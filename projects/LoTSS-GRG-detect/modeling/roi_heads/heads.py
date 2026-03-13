from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.structures import Instances


class MembershipHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, K) -> logits: (N, C)
        return self.net(x).squeeze(-1)


class ProposalValidityHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, K) proposal-level context vector -> logits: (N,)
        if x.dim() != 2:
            raise ValueError(
                f"ProposalValidityHead expects (N, K), got shape {tuple(x.shape)}"
            )
        return self.net(x).squeeze(-1)


class PhysicsAwareHeads(nn.Module):
    """
    Standalone physics-aware heads (decoupled from detectron2 ROI_HEADS_REGISTRY).

    Input contract:
      fused_features: Tensor with shape (B, P, C, K) or (N, C, K)
    where:
      B=batch, P=proposals/image, C=components/proposal, K=fused feature dim (F+D).

    Proposal contract (optional fields):
      - component_mask: (Pi, C) bool/0-1
      - gt_component_membership: (Pi, C) 0/1
      - gt_proposal_validity: (Pi,) 0/1
    """

    def __init__(
        self,
        input_dim: int,
        membership_head_hidden_dim: int = 256,
        proposal_head_hidden_dim: int = 256,
        membership_loss_weight: float = 1.0,
        proposal_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.membership_loss_weight = membership_loss_weight
        self.proposal_loss_weight = proposal_loss_weight
        self.membership_head = MembershipHead(
            input_dim=input_dim,
            hidden_dim=membership_head_hidden_dim,
        )
        self.proposal_validity_head = ProposalValidityHead(
            input_dim=input_dim,
            hidden_dim=proposal_head_hidden_dim,
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

        gt_membership = torch.cat(
            [self._to_tensor(inst.gt_component_membership, membership_logits.device) for inst in supervision],
            dim=0,
        ).float()

        if supervision[0].has("gt_proposal_validity"):
            gt_validity = torch.cat(
                [self._to_tensor(inst.gt_proposal_validity, proposal_validity_logits.device) for inst in supervision],
                dim=0,
            ).float()
        else:
            gt_validity = torch.ones_like(proposal_validity_logits)

        member_loss = F.binary_cross_entropy_with_logits(
            membership_logits,
            gt_membership,
            reduction="none",
        )
        member_loss = (member_loss * component_mask.float()).sum() / component_mask.float().sum().clamp_min(1.0)

        proposal_loss = F.binary_cross_entropy_with_logits(
            proposal_validity_logits,
            gt_validity,
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
        x = self._normalize_input(fused_features)  # (N, C, K)

        component_mask = self._get_component_mask(
            proposals=proposals,
            targets=targets,
            shape=x.shape[:2],
            device=x.device,
        )

        membership_logits = self.membership_head(x)
        membership_prob = torch.sigmoid(membership_logits) * component_mask.float()

        # Build a proposal-level context vector by a masked, membership-weighted
        # aggregation over components. This keeps a fixed-size representation (N, K)
        # and avoids mixing information across proposals.
        denom = membership_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        proposal_context = (x * membership_prob.unsqueeze(-1)).sum(dim=1) / denom

        proposal_validity_logits = self.proposal_validity_head(proposal_context)

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
    membership_head_hidden_dim = cfg.MODEL.MEMBERSHIP_HEAD.HIDDEN_DIM
    proposal_head_hidden_dim = cfg.MODEL.VALIDITY_HEAD.HIDDEN_DIM
    membership_loss_weight = cfg.MODEL.MEMBERSHIP_HEAD.LOSS_WEIGHT
    proposal_loss_weight = cfg.MODEL.VALIDITY_HEAD.LOSS_WEIGHT
    return PhysicsAwareHeads(
        input_dim=input_dim,
        membership_head_hidden_dim=membership_head_hidden_dim,
        proposal_head_hidden_dim=proposal_head_hidden_dim,
        membership_loss_weight=membership_loss_weight,
        proposal_loss_weight=proposal_loss_weight,
    )
