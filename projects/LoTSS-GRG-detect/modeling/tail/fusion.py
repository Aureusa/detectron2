import torch
import torch.nn as nn

from ..vanila import MHAttention


class AttentionFusionModule(nn.Module):
    """An attention-based fusion module to combine ROI features and the PhysicsFAN features."""
    def __init__(self, roi_feature_dim, physics_fan_feature_dim, dropout=0.0, num_heads=8):
        super().__init__()
        self.dim_align = nn.Linear(roi_feature_dim, physics_fan_feature_dim)
        self.attention = MHAttention(physics_fan_feature_dim, num_heads, dropout)

    def forward(self, roi_features, physics_fan_features):
        # Align the dimensions of ROI features to match PhysicsFAN features
        aligned_roi_features = self.dim_align(roi_features) # (B, P, physics_fan_feature_dim)

        physics_feats, membership = self._unpack_physics_fan_features(
            physics_fan_features
        )  # (B, P, C, physics_fan_feature_dim), (B, P, C)

        B, P, C, D = physics_feats.shape

        # Here the Q/K/V are as follows:
        # Q: ROI features (aligned to physics_fan_feature_dim)
        # K: PhysicsFAN features
        # V: PhysicsFAN features (V=K)
        query = aligned_roi_features.reshape(B * P, 1, D) # (B*P, 1, physics_fan_feature_dim)
        key = physics_feats.reshape(B * P, C, D) # (B*P, C, physics_fan_feature_dim)
        membership = membership.reshape(B * P, C) # (B*P, C)

        attn_output, attn_scores = self.attention(
            query, key, key,
            key_padding_mask=~membership.bool() # Mask out components not in the proposal
        ) # (B*P, 1, physics_fan_feature_dim)
        
        # Reshape attn_output back to (B, P, physics_fan_feature_dim)
        attn_output = attn_output.squeeze(1)  # (B*P, D)
        attn_output = attn_output.reshape(B, P, D) # (B, P, physics_fan_feature_dim)
        return attn_output, attn_scores
    
    def _unpack_physics_fan_features(self, physics_fan_features):
        physics_attended_features = physics_fan_features["attention_features"] # (B, P, C, physics_fan_feature_dim)
        membership_matrix = physics_fan_features["membership_matrix"] # (B, P, C)
        return physics_attended_features, membership_matrix
    

class ConcatenationFusionModule(nn.Module):
    """A simple fusion module that concatenates ROI features and PhysicsFAN features."""
    def __init__(self):
        super().__init__()

    def forward(self, roi_features, physics_fan_features):
        physics_feats = physics_fan_features["attention_features"] # (B, P, C, physics_fan_feature_dim)

        # ROI features: (B, P, roi_feature_dim)
        # PhysicsFAN features: (B, P, C, physics_fan_feature_dim)
        # We will concatenate along the feature dimension, so we need to reshape ROI features to match the component dimension
        B, P, roi_dim = roi_features.shape
        _, _, C, physics_dim = physics_feats.shape
        roi_features_expanded = roi_features.unsqueeze(2).expand(B, P, C, roi_dim) # (B, P, C, roi_feature_dim)
        fused_features = torch.cat([roi_features_expanded, physics_feats], dim=-1) # (B, P, C, roi_feature_dim + physics_fan_feature_dim)
        return fused_features
    

def build_fusion_module(cfg, roi_feature_dim, physics_fan_feature_dim):
    if cfg.MODEL.FUSION_MODULE.TYPE == "AttentionFusionModule":
        return AttentionFusionModule(
            roi_feature_dim=roi_feature_dim,
            physics_fan_feature_dim=physics_fan_feature_dim,
            dropout=cfg.MODEL.FUSION_MODULE.DROPOUT,
            num_heads=cfg.MODEL.FUSION_MODULE.NUM_HEADS,
        )
    elif cfg.MODEL.FUSION_MODULE.TYPE == "Concatenation":
        return ConcatenationFusionModule()
    else:
        raise ValueError(f"Unknown fusion module type: {cfg.MODEL.FUSION_MODULE.TYPE}. Supported types: 'AttentionFusionModule', 'Concatenation'")