import torch
from torch import nn

from .mlp import MLP


class ComponentAttention(nn.Module):
    """
    Self-attention over the component axis for tensors shaped (N, C, D).

    Architecture (pre-norm):
        LayerNorm
        MultiheadAttention (batch_first=True)
        + residual
        LayerNorm
        (no inner FFN — pair with ResidualMLPBlock if needed)
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.out_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, D)
            key_padding_mask: (N, C) bool — True means *ignore* that component.
        Returns:
            (N, C, D)
        """
        if x.dim() != 3:
            raise ValueError(f"ComponentAttention expects (N, C, D), got {tuple(x.shape)}")

        if key_padding_mask is not None:
            # Guard: avoid all-masked rows (produces NaN in softmax)
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.out_drop(x)
        return x + residual


class MHAttention(nn.Module):
    """Multi-Head Attention module for enhancing features with attention mechanisms."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(
            input_dim=embed_dim,
            hidden_dim=embed_dim * 4,  # common transformer ratio
            output_dim=embed_dim
        )

    def forward(self, query, key, value, key_padding_mask=None, output_mask=None):
        """
        Supports either 3D tensors (N, L, D) or 4D tensors (B, P, L, D).
        For 4D inputs we flatten (B, P) into N=B*P for nn.MultiheadAttention,
        then reshape outputs back to 4D.
        """
        original_query = query
        reshape_ctx = None

        if query.dim() == 4:
            if key.dim() != 4 or value.dim() != 4:
                raise ValueError("When query is 4D, key and value must also be 4D.")

            bsz, proposals, q_len, dim = query.shape
            _, _, k_len, _ = key.shape
            reshape_ctx = (bsz, proposals, q_len, k_len, dim)

            query = query.reshape(bsz * proposals, q_len, dim)
            key = key.reshape(bsz * proposals, k_len, dim)
            value = value.reshape(bsz * proposals, k_len, dim)

            if key_padding_mask is not None:
                if key_padding_mask.dim() == 3:
                    key_padding_mask = key_padding_mask.reshape(bsz * proposals, k_len)
                elif key_padding_mask.dim() != 2:
                    raise ValueError("key_padding_mask must be 2D or 3D.")

        elif query.dim() != 3:
            raise ValueError(
                f"query must be 3D or 4D, got shape {tuple(query.shape)}"
            )

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.bool()
            # Avoid rows where every key is masked (can produce NaNs in attention).
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask[all_masked, 0] = False

        mask_to_apply = None
        if output_mask is not None:
            output_mask = output_mask.bool()
            if reshape_ctx is None:
                if output_mask.dim() != 2:
                    raise ValueError("For 3D query, output_mask must be 2D (N, L).")
                if output_mask.shape != query.shape[:2]:
                    raise ValueError(
                        f"output_mask shape {tuple(output_mask.shape)} does not match query tokens {tuple(query.shape[:2])}."
                    )
                mask_to_apply = output_mask.unsqueeze(-1).float()
            else:
                bsz, proposals, q_len, _, _ = reshape_ctx
                if output_mask.dim() == 3:
                    output_mask = output_mask.reshape(bsz * proposals, q_len)
                elif output_mask.dim() != 2:
                    raise ValueError("For 4D query, output_mask must be 2D or 3D.")
                if output_mask.shape != query.shape[:2]:
                    raise ValueError(
                        f"output_mask shape {tuple(output_mask.shape)} does not match flattened query tokens {tuple(query.shape[:2])}."
                    )
                mask_to_apply = output_mask.unsqueeze(-1).float()

        attn_output, attn_score = self.attention(
            query, key, value, key_padding_mask=key_padding_mask
        )

        x = query + attn_output
        x = self.norm1(x)
        if mask_to_apply is not None:
            x = x * mask_to_apply
        x = x + self.mlp(x)
        x = self.norm2(x)
        if mask_to_apply is not None:
            x = x * mask_to_apply

        if reshape_ctx is not None:
            bsz, proposals, q_len, k_len, dim = reshape_ctx
            x = x.reshape(bsz, proposals, q_len, dim)
            if attn_score is not None and attn_score.dim() == 3:
                attn_score = attn_score.reshape(bsz, proposals, q_len, k_len)

            # Keep residual path semantics by returning in the original dimensionality.
            assert x.shape == original_query.shape

        return x, attn_score
    