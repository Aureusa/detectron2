import torch
from torch import nn


class ROIEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            
            nn.Conv2d(output_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C_in, H, W)
        Returns:
            (N, C_out, H, W)
        """
        if x.dim() != 4:
            raise ValueError(f"ROIEncoder expects (N, C_in, H, W), got {tuple(x.shape)}")
        return self.encoder(x)
    