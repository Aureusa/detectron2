import torch
from torch import nn

from .mlp import MLP


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

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, attn_score = self.attention(
            query, key, value, key_padding_mask=key_padding_mask
        )

        x = query + attn_output # Residual connection
        x = self.norm1(x)

        x = x + self.mlp(x) # Another residual connection
        x = self.norm2(x)

        return x, attn_score
    