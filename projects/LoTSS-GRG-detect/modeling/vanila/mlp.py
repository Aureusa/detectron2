import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    """
    Pre-norm residual MLP block.

    Shape: (..., H) -> (..., H)   (works for any leading dims)

    Architecture:
        LayerNorm
        Linear(H -> 4H)
        GELU
        Dropout
        Linear(4H -> H)
        + residual
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x + residual
    