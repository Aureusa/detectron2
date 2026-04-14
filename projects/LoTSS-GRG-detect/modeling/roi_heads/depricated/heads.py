from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.structures import Instances
from detectron2.layers import batched_nms

from ..vanila import ResidualMLPBlock


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
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            ResidualMLPBlock(hidden_dim, dropout=dropout),
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, input_dim) -> logits: (N, C)
        x = self.proj(x)          # (N, C, H)
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
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0, two_classes: bool = False):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            ResidualMLPBlock(hidden_dim, dropout=dropout),
        )
        # If two_classes is True, output 3 logits for multi-class classification (MCS/SCS/invalid)
        # instead of binary classification (valid/invalid)
        self.out = nn.Linear(hidden_dim, 1 if not two_classes else 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, input_dim) -> logits: (N,)
        if x.dim() != 2:
            raise ValueError(
                f"ProposalValidityHead expects (N, K), got shape {tuple(x.shape)}"
            )
        x = self.proj(x)    # (N, H)
        x = self.blocks(x)  # (N, H)
        return self.out(x).squeeze(-1)  # (N,) or (N, 3) if two_classes


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
        dropout: float = 0.0,
        membership_loss_weight: float = 1.0,
        proposal_loss_weight: float = 1.0,
        decouple_validity_projection: bool = True,
        two_classes: bool = False,
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
            dropout=dropout,
        )
        self.proposal_validity_head = ProposalValidityHead(
            input_dim=(hidden_dim * 3 + 1),
            hidden_dim=proposal_head_hidden_dim,
            dropout=dropout,
            two_classes=two_classes,
        )

        self.two_classes = two_classes

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

    def _membership_loss(
            self,
            membership_logits: torch.Tensor,
            component_mask: torch.Tensor,
            supervision: List[Instances]
        ) -> torch.Tensor:
        # Prevent a single unstable batch from poisoning BCE inputs.
        membership_logits = torch.nan_to_num(membership_logits, nan=0.0, posinf=30.0, neginf=-30.0)

        # Get the ground truth membership labels
        gt_membership = torch.cat(
            [self._to_tensor(inst.gt_component_membership, membership_logits.device) for inst in supervision],
            dim=0,
        ).float()
        gt_membership = torch.nan_to_num(gt_membership, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        total_valid = component_mask.float().sum().clamp_min(1.0)
        total_positive = (gt_membership * component_mask).float().sum()
        total_negative = total_valid - total_positive
        pos_weight = (total_negative / total_positive.clamp_min(1.0)).clamp(1, 10)

        # Compute binary cross-entropy loss for membership classification, masking out invalid components.
        member_loss = F.binary_cross_entropy_with_logits(
            membership_logits,
            gt_membership,
            pos_weight=pos_weight,
            reduction="none",
        )

        # NEW ---------------
        # valid_mask = component_mask.bool()

        # # --- HARD NEGATIVE MINING (PER PROPOSAL) ---
        # # Compute one membership loss per proposal, then average across proposals
        # # so large/easy proposals do not dominate global hard-negative selection.
        # proposal_losses = []
        # num_proposals = member_loss.shape[0]
        # for proposal_idx in range(num_proposals):
        #     proposal_loss = member_loss[proposal_idx]
        #     proposal_valid_mask = valid_mask[proposal_idx]
        #     proposal_gt = gt_membership[proposal_idx]

        #     if not proposal_valid_mask.any():
        #         continue

        #     proposal_pos_mask = (proposal_gt == 1) & proposal_valid_mask
        #     proposal_neg_mask = (proposal_gt == 0) & proposal_valid_mask

        #     proposal_pos_loss = proposal_loss[proposal_pos_mask]
        #     proposal_neg_loss = proposal_loss[proposal_neg_mask]

        #     if proposal_pos_loss.numel() > 0 and proposal_neg_loss.numel() > 0:
        #         k = min(proposal_neg_loss.numel(), proposal_pos_loss.numel() * 3)  # 1:3 ratio
        #         hard_neg_loss, _ = torch.topk(proposal_neg_loss, k)
        #         selected_loss = torch.cat([proposal_pos_loss, hard_neg_loss])
        #         proposal_losses.append(selected_loss.mean())
        #     elif proposal_pos_loss.numel() > 0:
        #         # Edge case: all valid labels are positive in this proposal.
        #         proposal_losses.append(proposal_pos_loss.mean())
        #     else:
        #         # No positives in this proposal: fall back to OLD masked-mean behavior.
        #         proposal_losses.append(proposal_loss[proposal_valid_mask].mean())

        # if len(proposal_losses) == 0:
        #     # Fully empty/invalid batch safety fallback.
        #     num_valid = valid_mask.float().sum().clamp_min(1.0)
        #     return (member_loss * valid_mask.float()).sum() / num_valid

        # return torch.stack(proposal_losses).mean()
    
        member_loss = (member_loss * component_mask.float()).sum() / component_mask.float().sum().clamp_min(1.0)
        return member_loss
    
    def _validity_loss(
            self,
            proposal_validity_logits: torch.Tensor,
            supervision: List[Instances]
        ) -> torch.Tensor:
        # Prevent a single unstable batch from poisoning Binary/Cross-Entropy inputs.
        proposal_validity_logits = torch.nan_to_num(
            proposal_validity_logits, nan=0.0, posinf=30.0, neginf=-30.0
        )

        gt_validity = torch.cat(
            [self._to_tensor(inst.gt_proposal_validity, proposal_validity_logits.device) for inst in supervision],
            dim=0,
        ).float()

        if self.two_classes:
            # Compute inverse frequency weights from this batch
            gt_validity_indices = torch.nan_to_num(
                gt_validity, nan=0.0, posinf=2.0, neginf=0.0
            ).long().clamp(0, 2)

            # Sqrt inverse frequency weights — softer than raw inverse frequency
            # Based on dataset statistics: invalid≈0.982, SCS≈0.016, MCS≈0.002
            raw_weights = torch.tensor(
                [1.0/0.982, 1.0/0.016, 1.0/0.002],
                device=proposal_validity_logits.device
            )
            fixed_weights = raw_weights.sqrt()
            fixed_weights = fixed_weights / fixed_weights.mean()

            proposal_loss = F.cross_entropy(
                proposal_validity_logits,
                gt_validity_indices,
                weight=fixed_weights,
            )
            # ---- OLD ----
            # gt_validity = torch.nan_to_num(
            #     gt_validity, nan=0.0, posinf=2.0, neginf=0.0
            # ).clamp(0.0, 2.0) # For two_classes, valid values are 0, 1, 2

            # gt_validity_indices = gt_validity.long()  # (N,) with values in {0, 1, 2}
            # proposal_loss = F.cross_entropy(
            #     proposal_validity_logits,
            #     gt_validity_indices,
            # )
        else:
            gt_validity = torch.nan_to_num(
            gt_validity, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0) # For two_classes, valid values are 0, 1

            proposal_loss = F.binary_cross_entropy_with_logits(
                proposal_validity_logits,
                gt_validity,
            )
        return proposal_loss

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
        if not supervision[0].has("gt_proposal_validity"):
            return {}

        return {
            "loss_membership": self._membership_loss(
                membership_logits=membership_logits,
                component_mask=component_mask,
                supervision=supervision,
            ) * self.membership_loss_weight,
            "loss_proposal_validity": self._validity_loss(
                proposal_validity_logits=proposal_validity_logits,
                supervision=supervision,
            ) * self.proposal_loss_weight,
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
        # two_classes=True: logits are (N, 3) → softmax so class probs sum to 1.
        # Stored as (N, 3); downstream use .argmax(dim=-1) for predicted class,
        # or inspect per-class probs directly for confusion/calibration analysis.
        if self.two_classes:
            validity_probs = torch.softmax(proposal_validity_logits, dim=-1)
        else:
            validity_probs = torch.sigmoid(proposal_validity_logits)

        start = 0
        for inst in proposals:
            n = len(inst)
            if n == 0:
                continue
            inst.pred_component_membership = membership_probs[start : start + n]
            inst.pred_proposal_validity = validity_probs[start : start + n]
            inst.pred_proposal_validity_logits = proposal_validity_logits[start : start + n]
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
            x_membership
        )  # (N, C)
        membership_logits = membership_logits.clamp(-30, 30) # Clamp logits before sigmoid to prevent NaNs in extreme cases.

        # Apply sigmoid to get membership probabilities, then mask out invalid components.
        # Also detach to prevent gradients from flowing into the membership
        # head when computing proposal losses.
        # TODO: consider not detaching if as right now the network does not use 
        # validity gradients to update the membership head, but this in theory is what we want.
        membership_prob = torch.sigmoid(membership_logits) * valid_mask  # (N, C)

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
                weighted_max_pool,
                # unweighted_mean, # Alternative: weighted_mean, TODO: try in the future if membership improvement is stale
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
    

class SetHeads(nn.Module):
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
        proposal_input_dim: int,
        membership_input_dim: int,
        membership_head_hidden_dim: int = 256,
        proposal_head_hidden_dim: int = 256,
        dropout: float = 0.0,
        membership_loss_weight: float = 1.0,
        proposal_loss_weight: float = 1.0,
        freeze_membership_head: bool = False,
        two_classes: bool = False,
    ):
        super().__init__()
        self.membership_loss_weight = membership_loss_weight
        self.proposal_loss_weight = proposal_loss_weight
        self.loss_card_weight = 0.05
        self.loss_sparse_weight = 0.01

        self.membership_head = MembershipHead(
            input_dim=membership_input_dim,
            hidden_dim=membership_head_hidden_dim,
            dropout=dropout,
        )
        self.proposal_validity_head = ProposalValidityHead(
            input_dim=proposal_input_dim,
            hidden_dim=proposal_head_hidden_dim,
            dropout=dropout,
            two_classes=two_classes,
        )

        self.contrastive_proj = nn.Sequential(
            nn.Linear(proposal_input_dim, proposal_input_dim),
            nn.GELU(),
            nn.Linear(proposal_input_dim, 128),
        )
        self.contrastive_loss_weight = 0.5
        self.temperature = 0.1

        self.two_classes = two_classes
        self.freeze_membership_head = freeze_membership_head

    def forward(
        self,
        enc_feats: torch.Tensor, # (N, C, K_enc)
        dec_feats: torch.Tensor, # (N, 1, K_dec)
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        membership_mask = self._get_membership_mask(
            proposals=proposals,
            targets=targets,
            shape=enc_feats.shape[:2],
            device=enc_feats.device,
        )  # (N, C) bool — True = valid component
        membership_mask = membership_mask.float()

        # ---- Compute proposal validity logits ----
        proposal_context = dec_feats.squeeze(1) # (N, K_dec)
        proposal_validity_logits = self.proposal_validity_head(proposal_context)  # (N,)

        # ---- Compute membership logits ----
        membership_logits = self.membership_head(
            enc_feats
        )  # (N, C)
        membership_logits = membership_logits.clamp(-30, 30) # Clamp logits before sigmoid to prevent NaNs in extreme cases.

        # ---- Compute contrastive proj ----
        contrastive_z = self.contrastive_proj(proposal_context) # (N, 128)
        contrastive_z = F.normalize(contrastive_z, dim=-1)

        if self.training:
            losses = self._compute_losses(
                membership_logits=membership_logits,
                proposal_validity_logits=proposal_validity_logits,
                contrastive_z=contrastive_z,
                membership_mask=membership_mask,
                proposals=proposals,
                targets=targets,
            )
            return proposals, losses

        outputs = self._attach_predictions(
            proposals=proposals,
            membership_logits=membership_logits,
            proposal_validity_logits=proposal_validity_logits,
            membership_mask=membership_mask,
        )
        return outputs, {}

    def _get_membership_mask(
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
    
    def _contrastive_loss(self, z: torch.Tensor, supervision: List[Instances]) -> torch.Tensor:
        gt_validity = torch.cat(
            [self._to_tensor(inst.gt_proposal_validity, z.device) for inst in supervision],
            dim=0,
        ).float()
        pos_mask = gt_validity.unsqueeze(1) * gt_validity.unsqueeze(0)  # (N, N) True if both proposals are valid

        # Remove self-comparisons
        self_mask = torch.eye(z.size(0), device=z.device)
        pos_mask = pos_mask * (1 - self_mask)

        sim = torch.matmul(z, z.T) / self.temperature  # (N, N) cosine similarity scaled by temperature
        exp_sim = torch.exp(sim) * (1 - self_mask)

        # For each anchor
        denom = exp_sim.sum(dim=1)  # (N,)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)  # (N,)

        anchor_mask = gt_validity > 0
        valid_anchors = (pos_sum > 0) & anchor_mask

        loss = -torch.log((pos_sum + 1e-8) / (denom + 1e-8))
        loss = loss[valid_anchors]

        if len(loss) == 0:
            return torch.tensor(0.0, device=z.device)

        return loss.mean()
    
    def _membership_loss(self, membership_logits, membership_mask, supervision):
        membership_logits = torch.nan_to_num(membership_logits, nan=0.0, posinf=30.0, neginf=-30.0)

        gt_membership = torch.cat(
            [self._to_tensor(inst.gt_component_membership, membership_logits.device) 
            for inst in supervision], dim=0,
        ).float()
        gt_membership = torch.nan_to_num(gt_membership, nan=0.0).clamp(0.0, 1.0)

        combined_mask = membership_mask.float()  # already restricted to valid proposals

        member_loss = F.binary_cross_entropy_with_logits(
            membership_logits,
            gt_membership,
            reduction="none",
        )  # (N, C)

        # Weight by proposal difficulty and mask
        loss = member_loss * combined_mask

        # Cardinality loss: predict the right NUMBER of members
        # Target is true member count, not component count
        gt_member_count = (gt_membership * combined_mask).sum(dim=1)  # (N,) — true members per proposal
        pred_member_count = membership_logits.sigmoid().sum(dim=1)    # (N,) — predicted member count
        L_card = F.l1_loss(pred_member_count, gt_member_count, reduction="none")
        # L_card = F.l1_loss(
        #     membership_logits.sigmoid().sum(dim=1),
        #     gt_membership,
        #     reduction="none",
        # )

        # Sparsitiy loss
        L_sparse = membership_logits.abs().mean(dim=1)
        return loss.sum() / combined_mask.sum().clamp_min(1.0), L_card.mean(), L_sparse.mean()

    def _validity_loss(
            self,
            proposal_validity_logits: torch.Tensor,
            supervision: List[Instances]
        ) -> torch.Tensor:
        # Prevent a single unstable batch from poisoning Binary/Cross-Entropy inputs.
        proposal_validity_logits = torch.nan_to_num(
            proposal_validity_logits, nan=0.0, posinf=30.0, neginf=-30.0
        )

        gt_validity = torch.cat(
            [self._to_tensor(inst.gt_proposal_validity, proposal_validity_logits.device) for inst in supervision],
            dim=0,
        ).float()

        if self.two_classes:
            gt_validity = torch.nan_to_num(
                gt_validity, nan=0.0, posinf=2.0, neginf=0.0
            ).clamp(0.0, 2.0) # For two_classes, valid values are 0, 1, 2

            gt_validity_indices = gt_validity.long()  # (N,) with values in {0, 1, 2}
            proposal_loss = F.cross_entropy(
                proposal_validity_logits,
                gt_validity_indices,
            )
        else:
            gt_validity = torch.nan_to_num(
            gt_validity, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0) # For two_classes, valid values are 0, 1

            proposal_loss = F.binary_cross_entropy_with_logits(
                proposal_validity_logits,
                gt_validity,
            )
        return proposal_loss, gt_validity

    def _compute_losses(
        self,
        membership_logits: torch.Tensor,
        proposal_validity_logits: torch.Tensor,
        contrastive_z: torch.Tensor,
        membership_mask: torch.Tensor,
        proposals: List[Instances],
        targets: Optional[List[Instances]],
    ) -> Dict[str, torch.Tensor]:
        supervision = targets if (targets is not None and len(targets) > 0) else proposals

        if len(supervision) == 0:
            return {}
        if not supervision[0].has("gt_component_membership"):
            return {}
        if not supervision[0].has("gt_proposal_validity"):
            return {}

        validity_loss, gt_validity = self._validity_loss(
            proposal_validity_logits=proposal_validity_logits,
            supervision=supervision,
        )

        contrastive_z_loss = self._contrastive_loss(
            z=contrastive_z,
            supervision=supervision,
        )

        # Couple the membership head through supervision:
        # if a proposal is invalid, its components should not be members; if valid, they should be.
        if not self.freeze_membership_head:
            valid_proposal_mask = (gt_validity > 0).unsqueeze(1)  # (N, 1) bool
            combined_mask = membership_mask.float() * valid_proposal_mask.float()  # (N, C) bool

            membership_loss, loss_card, loss_sparse = self._membership_loss(
                membership_logits=membership_logits,
                membership_mask=combined_mask,
                supervision=supervision,
            )

        losses = {
            "loss_proposal_validity": validity_loss * self.proposal_loss_weight,
            "loss_contrastive": contrastive_z_loss * self.contrastive_loss_weight,
        }

        if not self.freeze_membership_head:
            losses["loss_membership"] = membership_loss * self.membership_loss_weight
            losses["loss_card"] = loss_card * self.loss_card_weight
            losses["loss_sparse"] = loss_sparse * self.loss_sparse_weight
        return losses
    
    @staticmethod
    def _to_tensor(value, device: torch.device) -> torch.Tensor:
        if hasattr(value, "tensor"):
            return value.tensor.to(device)
        if isinstance(value, torch.Tensor):
            return value.to(device)
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    # def _attach_predictions(
    #     self,
    #     proposals: List[Instances],
    #     membership_logits: torch.Tensor,
    #     proposal_validity_logits: torch.Tensor,
    #     membership_mask: torch.Tensor,
    # ) -> List[Instances]:
    #     membership_probs = torch.sigmoid(membership_logits) # (N, C)

    #     # Mask out invalid components by setting their membership probs to 0.
    #     membership_probs = membership_probs * membership_mask

    #     # two_classes=True: logits are (N, 3) → softmax so class probs sum to 1.
    #     # Stored as (N, 3); downstream use .argmax(dim=-1) for predicted class,
    #     # or inspect per-class probs directly for confusion/calibration analysis.
    #     if self.two_classes:
    #         validity_probs = torch.softmax(proposal_validity_logits, dim=-1)
    #     else:
    #         validity_probs = torch.sigmoid(proposal_validity_logits)

    #     start = 0
    #     for inst in proposals:
    #         n = len(inst)
    #         if n == 0:
    #             continue
    #         inst.pred_component_membership = membership_probs[start : start + n]
    #         inst.pred_proposal_validity = validity_probs[start : start + n]
    #         inst.pred_proposal_validity_logits = proposal_validity_logits[start : start + n]
    #         start += n
    #     return proposals

    # TODO: Possible post processing step to collect IoU proposals
    def _attach_predictions(
        self,
        proposals: List[Instances],
        membership_logits: torch.Tensor,
        proposal_validity_logits: torch.Tensor,
        membership_mask: torch.Tensor,
    ) -> List[Instances]:
        membership_probs = torch.sigmoid(membership_logits)  # (N, C)
        membership_probs = membership_probs * membership_mask

        if self.two_classes:
            validity_probs = torch.softmax(proposal_validity_logits, dim=-1)  # (N, 3)
            nms_scores = validity_probs[:, 1:].max(dim=-1).values  # (N,) — max non-invalid prob
        else:
            validity_probs = torch.sigmoid(proposal_validity_logits)  # (N,)
            nms_scores = validity_probs  # (N,)

        result_proposals = []
        start = 0

        for inst in proposals:
            n = len(inst)
            if n == 0:
                result_proposals.append(inst)
                continue

            inst_membership = membership_probs[start:start + n]     # (n, C)
            inst_validity = validity_probs[start:start + n]         # (n,) or (n, 3)
            inst_logits = proposal_validity_logits[start:start + n] # (n,) or (n, 3)
            inst_scores = nms_scores[start:start + n]               # (n,)

            boxes = inst.proposal_boxes.tensor  # (n, 4)
            labels = torch.zeros(n, dtype=torch.long, device=boxes.device)

            keep = batched_nms(
                boxes,
                inst_scores,
                labels,
                iou_threshold=0.5,
            )

            # Build a boolean mask of which proposals survive NMS
            kept_mask = torch.zeros(n, dtype=torch.bool, device=boxes.device)
            kept_mask[keep] = True

            # Suppress non-kept proposals by marking them as invalid
            # This preserves index alignment with GT for the evaluator
            suppressed_membership = inst_membership.clone()
            suppressed_membership[~kept_mask] = 0.0

            if self.two_classes:
                suppressed_validity = inst_validity.clone()  # (n, 3)
                # Set suppressed proposals to class 0 (invalid) with full confidence
                suppressed_validity[~kept_mask] = torch.tensor(
                    [1.0, 0.0, 0.0],
                    device=inst_validity.device,
                )
            else:
                suppressed_validity = inst_validity.clone()  # (n,)
                suppressed_validity[~kept_mask] = 0.0

            suppressed_logits = inst_logits.clone()
            if self.two_classes:
                # Set logits so that softmax gives [1, 0, 0] — large negative for classes 1 and 2
                suppressed_logits[~kept_mask] = torch.tensor(
                    [10.0, -10.0, -10.0],
                    device=inst_logits.device,
                )
            else:
                suppressed_logits[~kept_mask] = -10.0  # sigmoid(-10) ≈ 0

            inst.pred_component_membership = suppressed_membership
            inst.pred_proposal_validity = suppressed_validity
            inst.pred_proposal_validity_logits = suppressed_logits

            result_proposals.append(inst)
            start += n

        return result_proposals


def build_physics_heads(cfg, input_dim) -> PhysicsAwareHeads:
    hidden_dim = cfg.MODEL.HEADS.HIDDEN_DIM
    membership_head_hidden_dim = cfg.MODEL.MEMBERSHIP_HEAD.HIDDEN_DIM
    proposal_head_hidden_dim = cfg.MODEL.VALIDITY_HEAD.HIDDEN_DIM
    membership_loss_weight = cfg.MODEL.MEMBERSHIP_HEAD.LOSS_WEIGHT
    proposal_loss_weight = cfg.MODEL.VALIDITY_HEAD.LOSS_WEIGHT
    dropout = cfg.MODEL.MEMBERSHIP_HEAD.DROPOUT
    decouple_validity_projection = cfg.MODEL.VALIDITY_HEAD.DECOUPLE_PROJECTION
    two_classes = cfg.MODEL.VALIDITY_HEAD.TWO_CLASSES # Whether to treat membership as binary (member vs non-member) or multi-class (MCS/SCS/invalid)
    return PhysicsAwareHeads(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        membership_head_hidden_dim=membership_head_hidden_dim,
        proposal_head_hidden_dim=proposal_head_hidden_dim,
        dropout=dropout,
        membership_loss_weight=membership_loss_weight,
        proposal_loss_weight=proposal_loss_weight,
        decouple_validity_projection=decouple_validity_projection,
        two_classes=two_classes,
    )

def build_set_heads(cfg, proposal_input_dim, membership_input_dim) -> SetHeads:
    membership_head_hidden_dim = cfg.MODEL.MEMBERSHIP_HEAD.HIDDEN_DIM
    proposal_head_hidden_dim = cfg.MODEL.VALIDITY_HEAD.HIDDEN_DIM
    membership_loss_weight = cfg.MODEL.MEMBERSHIP_HEAD.LOSS_WEIGHT
    proposal_loss_weight = cfg.MODEL.VALIDITY_HEAD.LOSS_WEIGHT
    dropout = cfg.MODEL.MEMBERSHIP_HEAD.DROPOUT
    two_classes = cfg.MODEL.VALIDITY_HEAD.TWO_CLASSES # Whether to treat membership as binary (member vs non-member) or multi-class (MCS/SCS/invalid)
    return SetHeads(
        proposal_input_dim=proposal_input_dim,
        membership_input_dim=membership_input_dim,
        membership_head_hidden_dim=membership_head_hidden_dim,
        proposal_head_hidden_dim=proposal_head_hidden_dim,
        dropout=dropout,
        membership_loss_weight=membership_loss_weight,
        proposal_loss_weight=proposal_loss_weight,
        two_classes=two_classes,
    )
