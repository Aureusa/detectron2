import torch.nn as nn

from detectron2.modeling.poolers import ROIPooler


class ROIALign(nn.Module):
    """Custom ROI Align module that wraps Detectron2's ROIPooler."""
    def __init__(self, pooler_resolution, pooler_scales, sampling_ratio, pooler_type):
        """
        Initializes the ROIALign module with the specified parameters.
        
        :param pooler_resolution: The output resolution (height and width) for the pooled features.
        :param pooler_scales: A tuple of scaling factors for each input feature map level
        :param sampling_ratio: The number of sampling points in the interpolation grid used to compute the output value of each pooled output bin.
        :param pooler_type: The type of pooling operation to use (e.g., "ROIAlignV2").
        """
        super().__init__()
        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        in_features       = cfg.MODEL.ROI_ALIGN.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_ALIGN.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_ALIGN.get("POOLER_SAMPLING_RATIO", 0)
        pooler_type       = cfg.MODEL.ROI_ALIGN.POOLER_TYPE
        return {
            "pooler_resolution": pooler_resolution,
            "pooler_scales": pooler_scales,
            "sampling_ratio": sampling_ratio,
            "pooler_type": pooler_type,
        }
    
    def forward(self, features, proposals):
        """
        Forward method to perform ROI Align on the input features based on the proposals.

        :param features: list[Tensor] or dict[str, Tensor] of backbone feature maps.
        :param proposals: list[Instances] or list[Boxes] for each image in the batch.
        """
        if isinstance(features, dict):
            feature_list = [features[f] for f in self.in_features]
        else:
            feature_list = features

        if len(proposals) > 0 and hasattr(proposals[0], "proposal_boxes"):
            boxes = [p.proposal_boxes for p in proposals]
        else:
            boxes = proposals

        return self.box_pooler(feature_list, boxes)
    
    def output_shape(self):
        return self.box_pooler.output_size
        

def build_roi_align(cfg, input_shape):
    kwargs = ROIALign.from_config(cfg, input_shape)
    module = ROIALign(**kwargs)
    module.in_features = cfg.MODEL.ROI_ALIGN.IN_FEATURES
    return module
