import torch
import torch.nn as nn

from ..vanila import TransformerBlock, ROIEncoder


class AttentionFusionModule(nn.Module):
    """
    Cross-attention fusion where each physics component queries the ROI spatial tokens.

    Q = physics component embeddings  (B*P, C_comp, D)
    K = V = ROI spatial patch tokens  (B*P, H*W, D)

    This is strictly intra-proposal: proposals are batched as B*P so attention
    never crosses proposal boundaries. No pooling is done, so the full spatial
    detail of each RoI map is available to each component.

    Output: (B, P, C_comp, D) — same shape as PhysicsFAN output, suitable for
    the per-component membership/validity heads.
    """

    def __init__(self, roi_feature_dim, physics_fan_feature_dim, dropout=0.0, num_heads=8, bidirectional=False):
        super().__init__()
        # --- ROI spatial tokens ---
        # Project ROI channels to physics embedding dim so Q and K share the same space.
        self.roi_encoder = ROIEncoder(roi_feature_dim, roi_feature_dim, dropout=dropout)
        self.roi_projector = nn.Sequential(
            nn.LayerNorm(roi_feature_dim),
            nn.Linear(roi_feature_dim, physics_fan_feature_dim)
        )

        # --- Cross-attention blocks ---
        self.transformer_block = TransformerBlock(
            embed_dim=physics_fan_feature_dim, num_heads=num_heads, dropout=dropout
        )

        if bidirectional:
            self.transformer_block_roi = TransformerBlock(
                embed_dim=physics_fan_feature_dim, num_heads=num_heads, dropout=dropout
            )

            # --- Fusion ---
            self.fusion_gate = nn.Sequential(
                nn.Linear(physics_fan_feature_dim * 2, physics_fan_feature_dim),
                nn.Sigmoid(),
            )
            self.fusion_context_proj = nn.Linear(physics_fan_feature_dim, physics_fan_feature_dim)

            self.fusion_norm = nn.LayerNorm(physics_fan_feature_dim)

        self.bidirectional = bidirectional

    def forward(self, roi_features, physics_fan_features):
        # Unpack packed features from PhysicsFAN
        physics_feats, membership = self._unpack_physics_fan_features(
            physics_fan_features
        )  # (B, P, C_comp, D), (B, P, C_comp)
        physics_fan_spatial_feats = self._unpack_physics_fan_spatial_features(
            physics_fan_features
        )  # (B, P, C_comp, 2)

        bsz, proposals_per_image, num_components, feat_dim = physics_feats.shape
        expected_n = bsz * proposals_per_image

        # --- Build ROI spatial tokens (no pooling) ---
        # roi_features: (N, roi_dim, H, W)  with N == B*P
        if roi_features.dim() != 4 or roi_features.shape[0] != expected_n:
            raise ValueError(
                f"Expected roi_features shape (B*P, C, H, W)=({expected_n}, *, *, *), "
                f"got {tuple(roi_features.shape)}."
            )
        n, c, h, w = roi_features.shape
        roi_features = self.roi_encoder(roi_features) # (N, roi_dim, H, W)
        roi_features = self.roi_projector(roi_features.permute(0, 2, 3, 1))  # (N, H, W, D)

        # (N, H, W, D) -> (N, H*W, D)
        roi_spatial = roi_features.reshape(n, h * w, feat_dim)

        # --- ADD CLS TOKEN ---
        # Use mean-pooled token as CLS since it provides a global summary of the RoI
        cls_token = roi_spatial.mean(dim=1, keepdim=True)  # (N, 1, C)
        roi_spatial = torch.cat([cls_token, roi_spatial], dim=1)  # (N, 1 + H*W, C)

        # Add positional encoding ase we are using roi_spatial as keys/values in attention.
        # This helps the model learn spatial patterns.
        pos = self._build_2d_sincos_position_embedding(
            h, w, feat_dim, device=roi_features.device
        )

        # Add positional encoding to physics features as well,
        # since they are queries in attention and need spatial info to attend effectively.
        spatial_pos = self._build_component_position_encoding(
            physics_fan_spatial_feats[..., 0],  # dx
            physics_fan_spatial_feats[..., 1],  # dy
            membership,  # (B, P, C)
            h, w, feat_dim, device=physics_feats.device
        )  # (B, P, C, D)
        physics_feats = physics_feats + spatial_pos  # (B, P, C, D)

        # IMPORTANT: expand pos to match CLS
        cls_pos = torch.zeros(1, feat_dim, device=roi_features.device)  # (1, C)
        pos = torch.cat([cls_pos, pos], dim=0)  # (1 + H*W, C)

        roi_spatial = roi_spatial + pos.unsqueeze(0)  # (N, 1 + H*W, D)
        if self.bidirectional:
            return self._bidirectional_fusion(
                physics_feats,
                roi_spatial,
                membership,
                expected_n,
                num_components,
                feat_dim,
                bsz,
                proposals_per_image,
            )
        else:
            return self._unidirectional_fusion(
                physics_feats,
                roi_spatial,
                membership,
                expected_n,
                num_components,
                feat_dim,
                bsz,
                proposals_per_image,
            )

    def _build_component_position_encoding(
            self,
            scaled_dx,   # (B, P, C) — already scaled by proposal width, range [-0.5, 0.5]
            scaled_dy,   # (B, P, C) — already scaled by proposal height, range [-0.5, 0.5]
            membership,  # (B, P, C)
            h, w,
            feat_dim,
            device
        ):
        bsz, proposals_per_image, num_components = scaled_dx.shape

        # Turn into (B, P, C) -> (N, C) where N=B*P for easier processing
        scaled_dx = scaled_dx.reshape(-1, scaled_dx.shape[2])  # (N, C)
        scaled_dy = scaled_dy.reshape(-1, scaled_dy.shape[2])  # (N, C)

        # Shift from [-0.5, 0.5] to [0, 1]
        norm_x = (scaled_dx + 0.5).clamp(0.0, 1.0)  # (N, C)
        norm_y = (scaled_dy + 0.5).clamp(0.0, 1.0)  # (N, C)

        # Map to grid indices [0, W-1] and [0, H-1]
        grid_x = norm_x * (w - 1)  # (N, C)
        grid_y = norm_y * (h - 1)  # (N, C)

        # Split the embedding budget across x/y so both coordinates are represented.
        x_dim = feat_dim // 2
        y_dim = feat_dim - x_dim

        x_freqs = max(1, (x_dim + 1) // 2)
        y_freqs = max(1, (y_dim + 1) // 2)

        omega_x = torch.arange(x_freqs, device=device, dtype=grid_x.dtype) / x_freqs
        omega_y = torch.arange(y_freqs, device=device, dtype=grid_y.dtype) / y_freqs
        omega_x = 1.0 / (10000 ** omega_x)
        omega_y = 1.0 / (10000 ** omega_y)

        out_x = grid_x.reshape(-1, 1) * omega_x  # (N*C, x_freqs)
        out_y = grid_y.reshape(-1, 1) * omega_y  # (N*C, y_freqs)

        pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)[:, :x_dim]  # (N*C, x_dim)
        pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)[:, :y_dim]  # (N*C, y_dim)

        pos = torch.cat([pos_x, pos_y], dim=1)  # (N*C, feat_dim)
        pos = pos.reshape(scaled_dx.shape[0], scaled_dx.shape[1], feat_dim)  # (N, C, feat_dim)

        # Since we rescaled components outside the proposals will be at the center of
        # the proposal [0.5, 0.5]. We mask those as they are NOT valid components.
        mask = membership.reshape(-1, membership.shape[2])  # (N, C)
        pos = pos * mask.unsqueeze(-1).float()  # Mask out invalid components (N, C, feat_dim)
        return pos.reshape(bsz, proposals_per_image, num_components, feat_dim)  # (B, P, C, feat_dim)

    def _unidirectional_fusion(
            self,
            physics_feats,
            roi_spatial,
            membership,
            expected_n,
            num_components,
            feat_dim,
            bsz,
            proposals_per_image
        ):
        # --- Cross-attention: component queries attend to spatial ROI keys ---
        # Q: (N, C_comp, D)  K=V: (N, 1 + H*W, D)
        query = physics_feats.reshape(expected_n, num_components, feat_dim)
        key = roi_spatial  # (N, 1 + H*W, D)

        physics_attends_to_roi, _ = self.transformer_block(
            query, key, key
            # No key_padding_mask needed: all spatial positions are valid.
        )  # (N, C_comp, D)

        fused = physics_attends_to_roi.reshape(bsz, proposals_per_image, num_components, feat_dim) # (B, P, C_comp, D)
        # fused = fused * membership.unsqueeze(-1).float()  # Mask out components not in the proposal
        return (
            fused, # (B, P, C_comp, D)
            {
                "physics_to_roi": physics_attends_to_roi, # (N, C_comp, D)
                "membership": membership # (B, P, C)
            }
        )
    
    def _bidirectional_fusion(
            self,
            physics_feats,
            roi_spatial,
            membership,
            expected_n,
            num_components,
            feat_dim,
            bsz,
            proposals_per_image
        ):
        # --- Cross-attention: component queries attend to spatial ROI keys ---
        # Q: (N, C_comp, D)  K=V: (N, 1 + H*W, D)
        query = physics_feats.reshape(expected_n, num_components, feat_dim)
        key = roi_spatial  # (N, 1 + H*W, D)

        physics_attends_to_roi, _ = self.transformer_block(
            query, key, key
            # No key_padding_mask needed: all spatial positions are valid.
        )  # (N, C_comp, D)

        # --- Cross-attention: ROI queries attend to component keys ---
        # Q: (N, 1 + H*W, D)  K=V: (N, C_comp, D)
        query = roi_spatial  # (N, 1 + H*W, D)
        key = physics_feats.reshape(expected_n, num_components, feat_dim)

        roi_attends_to_physics, _ = self.transformer_block_roi(
            query, key, key,
            # We need mask as some proposals have fewer components, so some rows in physics_feats are padding.
            key_padding_mask=~membership.reshape(expected_n, num_components).bool()  # (N, C_comp)
        )   # (N, 1 + H*W, D)

        # --- Broadcasting ---
        # Reshape back to (B, P, C_comp, D)
        physics_attends_to_roi = physics_attends_to_roi.reshape(
            bsz, proposals_per_image, num_components, feat_dim
        )   # (B, P, C_comp, D)

        # Also reshape roi_attends_to_physics to (B, P, 1 + H*W, D)
        roi_attends_to_physics = roi_attends_to_physics.reshape(
            bsz, proposals_per_image, -1, feat_dim
        )   # (B, P, 1 + H*W, D)

        # --- Fusion ---
        # Use CLS token as proposal-level ROI context, then gate it into component space.
        roi_context = roi_attends_to_physics[:, :, 0, :]  # (B, P, D)
        roi_context = roi_context.unsqueeze(2).expand(-1, -1, num_components, -1)  # (B, P, C_comp, D)

        gate_input = torch.cat([physics_attends_to_roi, roi_context], dim=-1)  # (B, P, C_comp, 2D)
        gate = self.fusion_gate(gate_input)  # (B, P, C_comp, D)
        delta = self.fusion_context_proj(roi_context)  # (B, P, C_comp, D)

        fused = self.fusion_norm(physics_attends_to_roi + gate * delta + physics_feats)
        fused = fused * membership.unsqueeze(-1).float()  # Mask out components not in the proposal

        return fused, {
            "physics_to_roi": physics_attends_to_roi,
            "roi_to_physics": roi_attends_to_physics,
            "gate": gate,
        }

    def _unpack_physics_fan_features(self, physics_fan_features):
        physics_attended_features = physics_fan_features["attention_features"]  # (B, P, C, D)
        membership_matrix = physics_fan_features["membership_matrix"]  # (B, P, C)
        return physics_attended_features, membership_matrix
    
    def _unpack_physics_fan_spatial_features(self, physics_fan_features):
        spatial_features = physics_fan_features["spatial_features"]  # (B, P, C, 2)
        return spatial_features
    
    def _build_2d_sincos_position_embedding(self, h, w, dim, device):
        """
        Build 2D sinusoidal positional embeddings.

        Returns: (H*W, dim)
        """
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij"
        )

        grid = torch.stack((grid_x, grid_y), dim=-1).float()  # (H, W, 2)

        # Split the embedding budget across x/y so both coordinates are represented.
        x_dim = dim // 2
        y_dim = dim - x_dim

        x_freqs = max(1, (x_dim + 1) // 2)
        y_freqs = max(1, (y_dim + 1) // 2)

        omega_x = torch.arange(x_freqs, device=device, dtype=grid.dtype) / x_freqs
        omega_y = torch.arange(y_freqs, device=device, dtype=grid.dtype) / y_freqs
        omega_x = 1.0 / (10000 ** omega_x)
        omega_y = 1.0 / (10000 ** omega_y)

        out_x = grid[..., 0].reshape(-1, 1) * omega_x
        out_y = grid[..., 1].reshape(-1, 1) * omega_y

        pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)[:, :x_dim]
        pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)[:, :y_dim]

        pos = torch.cat([pos_x, pos_y], dim=1)

        return pos


class ConcatenationFusionModule(nn.Module):
    """
    Concatenation fusion: for each component, concatenates the ROI spatially-averaged
    token (per-proposal) with the PhysicsFAN component embedding.

    Note: unlike AttentionFusionModule this does average-pool the spatial ROI map,
    since concatenation cannot operate across a sequence of H*W tokens per component.
    """

    def __init__(self, roi_feature_dim, physics_fan_feature_dim):
        super().__init__()
        self.roi_proj = nn.Linear(roi_feature_dim, physics_fan_feature_dim)

    def forward(self, roi_features, physics_fan_features):
        physics_feats = physics_fan_features["attention_features"]  # (B, P, C, D)
        bsz, proposals_per_image, num_components, _ = physics_feats.shape
        expected_n = bsz * proposals_per_image

        if roi_features.dim() != 4 or roi_features.shape[0] != expected_n:
            raise ValueError(
                f"Expected roi_features (B*P, C, H, W)=({expected_n}, *, *, *), "
                f"got {tuple(roi_features.shape)}."
            )
        # Pool to (N, roi_dim) then project to (B, P, D)
        roi_token = roi_features.mean(dim=[2, 3])  # (N, roi_dim)
        roi_token = self.roi_proj(roi_token)  # (N, D)
        roi_token = roi_token.view(bsz, proposals_per_image, 1, -1)  # (B, P, 1, D)
        roi_expanded = roi_token.expand(bsz, proposals_per_image, num_components, -1)  # (B, P, C, D)

        fused_features = torch.cat([roi_expanded, physics_feats], dim=-1)  # (B, P, C, 2D)
        return fused_features


def build_fusion_module(cfg, roi_feature_dim, physics_fan_feature_dim):
    if cfg.MODEL.FUSION_MODULE.TYPE == "AttentionFusionModule":
        return AttentionFusionModule(
            roi_feature_dim=roi_feature_dim,
            physics_fan_feature_dim=physics_fan_feature_dim,
            dropout=cfg.MODEL.FUSION_MODULE.DROPOUT,
            num_heads=cfg.MODEL.FUSION_MODULE.NUM_HEADS,
            bidirectional=cfg.MODEL.FUSION_MODULE.BIDIRECTIONAL
        )
    elif cfg.MODEL.FUSION_MODULE.TYPE == "Concatenation":
        return ConcatenationFusionModule(
            roi_feature_dim=roi_feature_dim,
            physics_fan_feature_dim=physics_fan_feature_dim,
        )
    else:
        raise ValueError(f"Unknown fusion module type: {cfg.MODEL.FUSION_MODULE.TYPE}. Supported types: 'AttentionFusionModule', 'Concatenation'")
    