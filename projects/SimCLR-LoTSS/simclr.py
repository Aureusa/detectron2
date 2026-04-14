import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRWithFPN(nn.Module):
    def __init__(self, backbone, projection_dim=128, feature_name="p5"):
        super().__init__()
        self.backbone = backbone
        self.feature_name = feature_name

        input_dim = self._get_feature_dim(feature_name)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def _get_feature_dim(self, feature_name):
        if hasattr(self.backbone, "output_shape"):
            output_shapes = self.backbone.output_shape()
            if feature_name in output_shapes:
                return output_shapes[feature_name].channels

        raise ValueError(
            f"Backbone does not expose feature '{feature_name}' in output_shape()."
        )

    def forward(self, x):
        # FPN returns a dict: {"p2": ..., "p3": ..., "p4": ..., "p5": ...}
        features = self.backbone(x)

        if self.feature_name not in features:
            raise KeyError(
                f"Feature '{self.feature_name}' not found in backbone outputs: {list(features.keys())}"
            )

        pooled = F.adaptive_avg_pool2d(features[self.feature_name], (1, 1)).flatten(1)
        z = self.projector(pooled)
        return F.normalize(z, dim=-1)
    