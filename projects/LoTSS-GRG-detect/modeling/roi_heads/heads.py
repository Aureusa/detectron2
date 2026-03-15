from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.structures import Instances

from ..vanila import ResidualMLPBlock, ComponentAttention


class MembershipHead(nn.Module):
    """
    Per-component membership classifier.

    Architecture:
        Linear(input_dim -> hidden_dim)
        ResidualMLPBlock x2
        ComponentAttention
        Linear(hidden_dim -> 1)

    Input:  (N, C, input_dim)   — shared-projected features
    Output: (N, C)              — per-component membership logits
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.attn = ComponentAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.blocks = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            ResidualMLPBlock(hidden_dim, dropout=dropout),
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (N, C, input_dim) -> logits: (N, C)
        x = self.proj(x)          # (N, C, H)
        x = self.attn(x, key_padding_mask=key_padding_mask)  # (N, C, H)
        x = self.blocks(x)        # (N, C, H)
        return self.out(x).squeeze(-1)  # (N, C)


class ProposalValidityHead(nn.Module):
    """
    Proposal-level validity classifier.

    Architecture:
        Linear(input_dim -> hidden_dim)
        ResidualMLPBlock x2
        Linear(hidden_dim -> 1)

    Input:  (N, input_dim)  — membership-weighted component context
    Output: (N,)            — proposal validity logits
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            ResidualMLPBlock(hidden_dim, dropout=dropout),
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, input_dim) -> logits: (N,)
        if x.dim() != 2:
            raise ValueError(
                f"ProposalValidityHead expects (N, K), got shape {tuple(x.shape)}"
            )
        x = self.proj(x)    # (N, H)
        x = self.blocks(x)  # (N, H)
        return self.out(x).squeeze(-1)  # (N,)


class PhysicsAwareHeads(nn.Module):
    """
    Standalone physics-aware heads (decoupled from detectron2 ROI_HEADS_REGISTRY).

    Pipeline:
        fused_features (N, C, input_dim)
            ↓  shared_proj  Linear(input_dim → hidden_dim)
        (N, C, hidden_dim)
            ↓  MembershipHead  (projection + 2x ResidualMLP + ComponentAttention + Linear)
        membership_logits (N, C)
            ↓  membership-weighted pooling
        proposal_context (N, hidden_dim)
            ↓  ProposalValidityHead  (projection + 2x ResidualMLP + Linear)
        proposal_validity_logits (N,)

    Input contract:
        fused_features: (B, P, C, K) or (N, C, K)
          B=batch  P=proposals/image  C=components/proposal  K=fused feature dim

    Proposal fields (optional):
        component_mask:          (Pi, C)  bool / 0-1
        gt_component_membership: (Pi, C)  0/1
        gt_proposal_validity:    (Pi,)    0/1
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        membership_head_hidden_dim: int = 256,
        proposal_head_hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
        membership_loss_weight: float = 1.0,
        proposal_loss_weight: float = 1.0,
        decouple_validity_projection: bool = True,
    ):
        super().__init__()
        self.membership_loss_weight = membership_loss_weight
        self.proposal_loss_weight = proposal_loss_weight
        self.decouple_validity_projection = decouple_validity_projection

        # Shared projection applied before both heads.
        self.shared_proj = nn.Linear(input_dim, hidden_dim)
        self.shared_drop = nn.Dropout(dropout)

        # Optional separate projection for validity features so validity gradients
        # do not dominate the shared membership projection parameters.
        self.validity_proj = nn.Linear(input_dim, hidden_dim)
        self.validity_drop = nn.Dropout(dropout)

        self.membership_head = MembershipHead(
            input_dim=hidden_dim,
            hidden_dim=membership_head_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.proposal_validity_head = ProposalValidityHead(
            input_dim=(hidden_dim * 3 + 1),
            hidden_dim=proposal_head_hidden_dim,
            dropout=dropout,
        )

    def _normalize_input(self, fused_features: torch.Tensor) -> torch.Tensor:
        if isinstance(fused_features, tuple):
            fused_features = fused_features[0]
        if fused_features.dim() == 4:
            bsz, proposals_per_image, num_components, feature_dim = fused_features.shape
            return fused_features.reshape(bsz * proposals_per_image, num_components, feature_dim)
        if fused_features.dim() == 3:
            return fused_features
        raise ValueError(
            f"Expected fused_features shape (B,P,C,K) or (N,C,K), got {tuple(fused_features.shape)}"
        )

    def _get_component_mask(
        self,
        proposals: List[Instances],
        targets: Optional[List[Instances]],
        shape: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if targets is not None and len(targets) > 0 and targets[0].has("component_mask"):
            return torch.cat([t.component_mask.to(device) for t in targets], dim=0).bool()
        if len(proposals) == 0:
            return torch.ones(shape, dtype=torch.bool, device=device)
        if proposals[0].has("component_mask"):
            return torch.cat([inst.component_mask.to(device) for inst in proposals], dim=0).bool()
        return torch.ones(shape, dtype=torch.bool, device=device)

    def _compute_losses(
        self,
        membership_logits: torch.Tensor,
        proposal_validity_logits: torch.Tensor,
        component_mask: torch.Tensor,
        proposals: List[Instances],
        targets: Optional[List[Instances]],
    ) -> Dict[str, torch.Tensor]:
        supervision = targets if (targets is not None and len(targets) > 0) else proposals

        if len(supervision) == 0:
            return {}
        if not supervision[0].has("gt_component_membership"):
            return {}

        # Prevent a single unstable batch from poisoning BCE inputs.
        membership_logits = torch.nan_to_num(membership_logits, nan=0.0, posinf=30.0, neginf=-30.0)
        proposal_validity_logits = torch.nan_to_num(
            proposal_validity_logits, nan=0.0, posinf=30.0, neginf=-30.0
        )

        gt_membership = torch.cat(
            [self._to_tensor(inst.gt_component_membership, membership_logits.device) for inst in supervision],
            dim=0,
        ).float()
        gt_membership = torch.nan_to_num(gt_membership, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        if supervision[0].has("gt_proposal_validity"):
            gt_validity = torch.cat(
                [self._to_tensor(inst.gt_proposal_validity, proposal_validity_logits.device) for inst in supervision],
                dim=0,
            ).float()
        else:
            gt_validity = torch.ones_like(proposal_validity_logits)
        gt_validity = torch.nan_to_num(gt_validity, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # For membership validity, use pos_weight to handle class imbalance
        # (many more invalid members than valid ones).
        pos_count = (gt_membership == 1).float().sum()
        neg_count = (gt_membership == 0).float().sum()
        pos_weight = (neg_count / pos_count.clamp_min(1.0)).clamp(max=20.0)
        
        member_loss = F.binary_cross_entropy_with_logits(
            membership_logits,
            gt_membership,
            pos_weight=pos_weight,
        )
        member_loss = (member_loss * component_mask.float()).sum() / component_mask.float().sum().clamp_min(1.0)

        # For proposal validity, use pos_weight to handle class imbalance
        # (many more invalid proposals than valid ones).
        pos_count = (gt_validity == 1).float().sum()
        neg_count = (gt_validity == 0).float().sum()
        pos_weight = (neg_count / pos_count.clamp_min(1.0)).clamp(max=20.0)

        proposal_loss = F.binary_cross_entropy_with_logits(
            proposal_validity_logits,
            gt_validity,
            pos_weight=pos_weight,
        )

        return {
            "loss_membership": member_loss * self.membership_loss_weight,
            "loss_proposal_validity": proposal_loss * self.proposal_loss_weight,
        }

    @staticmethod
    def _to_tensor(value, device: torch.device) -> torch.Tensor:
        if hasattr(value, "tensor"):
            return value.tensor.to(device)
        if isinstance(value, torch.Tensor):
            return value.to(device)
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    def _attach_predictions(
        self,
        proposals: List[Instances],
        membership_logits: torch.Tensor,
        proposal_validity_logits: torch.Tensor,
    ) -> List[Instances]:
        membership_probs = torch.sigmoid(membership_logits)
        validity_probs = torch.sigmoid(proposal_validity_logits)

        start = 0
        for inst in proposals:
            n = len(inst)
            if n == 0:
                continue
            inst.pred_component_membership = membership_probs[start : start + n]
            inst.pred_proposal_validity = validity_probs[start : start + n]
            start += n
        return proposals

    def forward(
        self,
        fused_features: torch.Tensor,
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        x_input = self._normalize_input(fused_features)  # (N, C, K)

        component_mask = self._get_component_mask(
            proposals=proposals,
            targets=targets,
            shape=x_input.shape[:2],
            device=x_input.device,
        )  # (N, C) bool — True = valid component
        valid_mask = component_mask.float()

        # Shared projection: (N, C, K) -> (N, C, H)
        x_membership = self.shared_drop(self.shared_proj(x_input))

        if self.decouple_validity_projection:
            x_validity = self.validity_drop(self.validity_proj(x_input))
        else:
            x_validity = x_membership

        # Membership head expects invalid-component mask for key_padding_mask
        # (True = ignore), so we invert the valid mask.
        membership_logits = self.membership_head(
            x_membership, key_padding_mask=~component_mask
        )  # (N, C)
        membership_logits = membership_logits.clamp(-30, 30) # Clamp logits before sigmoid to prevent NaNs in extreme cases.

        # Apply sigmoid to get membership probabilities, then mask out invalid components.
        # Also detach to prevent gradients from flowing into the membership
        # head when computing proposal losses.
        # TODO: consider not detaching if as right now the network does not use 
        # validity gradients to update the membership head, but this in theory is what we want.
        membership_prob = torch.sigmoid(membership_logits).detach() * valid_mask  # (N, C)

        # Build a richer proposal context for validity classification:
        # [membership-weighted mean, unweighted valid-component mean,
        #  valid-component max pool, valid-component count].
        weighted_denom = membership_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weighted_mean = (x_validity * membership_prob.unsqueeze(-1)).sum(dim=1) / weighted_denom

        raw_valid_count = valid_mask.sum(dim=1, keepdim=True)
        valid_count = raw_valid_count.clamp_min(1.0)
        unweighted_mean = (x_validity * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_count

        # Alternative weighted max. This substitutes unweighted mean since this gives the proposal
        # head a membership-free shortcut, which makes component-level supervision less effective.

        # Keep max pooling numerically stable by using finite fill values.
        x_valid_masked = x_validity.masked_fill(~component_mask.unsqueeze(-1), -1e9)
        max_pool = x_valid_masked.max(dim=1).values
        max_pool = torch.where(raw_valid_count > 0, max_pool, torch.zeros_like(max_pool))
        
        # Alternative weighted max. This substitutes unweighted mean since this gives the proposal
        # head a membership-free shortcut, which makes component-level supervision less effective.
        # Multiply the component features by the membership probabilities to get a weighted max
        # then mask out invalid components with a large negative value to prevent them from dominating the max.
        x_weighted = x_validity * membership_prob.unsqueeze(-1)
        x_weighted_masked = x_weighted.masked_fill(~component_mask.unsqueeze(-1), -1e9)
        weighted_max_pool = x_weighted_masked.max(dim=1).values
        weighted_max_pool = torch.where(raw_valid_count > 0, weighted_max_pool, torch.zeros_like(weighted_max_pool))

        proposal_context = torch.cat(
            [
                weighted_mean,
                unweighted_mean, # Alternative: weighted_mean, TODO: try in the future if membership improvement is stale
                max_pool,
                valid_count
            ], dim=1)
        proposal_context = torch.nan_to_num(proposal_context, nan=0.0, posinf=1e4, neginf=-1e4)

        proposal_validity_logits = self.proposal_validity_head(proposal_context)  # (N,)

        if self.training:
            losses = self._compute_losses(
                membership_logits=membership_logits,
                proposal_validity_logits=proposal_validity_logits,
                component_mask=component_mask,
                proposals=proposals,
                targets=targets,
            )
            return proposals, losses

        outputs = self._attach_predictions(
            proposals=proposals,
            membership_logits=membership_logits,
            proposal_validity_logits=proposal_validity_logits,
        )
        return outputs, {}


def build_physics_heads(cfg, input_dim) -> PhysicsAwareHeads:
    hidden_dim = cfg.MODEL.HEADS.HIDDEN_DIM
    membership_head_hidden_dim = cfg.MODEL.MEMBERSHIP_HEAD.HIDDEN_DIM
    proposal_head_hidden_dim = cfg.MODEL.VALIDITY_HEAD.HIDDEN_DIM
    membership_loss_weight = cfg.MODEL.MEMBERSHIP_HEAD.LOSS_WEIGHT
    proposal_loss_weight = cfg.MODEL.VALIDITY_HEAD.LOSS_WEIGHT
    num_heads = cfg.MODEL.MEMBERSHIP_HEAD.NUM_HEADS
    dropout = cfg.MODEL.MEMBERSHIP_HEAD.DROPOUT
    decouple_validity_projection = cfg.MODEL.VALIDITY_HEAD.DECOUPLE_PROJECTION
    return PhysicsAwareHeads(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        membership_head_hidden_dim=membership_head_hidden_dim,
        proposal_head_hidden_dim=proposal_head_hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        membership_loss_weight=membership_loss_weight,
        proposal_loss_weight=proposal_loss_weight,
        decouple_validity_projection=decouple_validity_projection,
    )
