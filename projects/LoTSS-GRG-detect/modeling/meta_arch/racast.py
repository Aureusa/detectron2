import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from ..set_transformer import build_set_transformer
from ..tail import build_physics_fn, build_fusion_module
from ..roi_heads import build_roi_align, build_set_heads


@META_ARCH_REGISTRY.register()
class RaCAST(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        physics_fn: nn.Module,
        fusion_module: nn.Module,
        roi_align: nn.Module,
        set_transformer: nn.Module,
        heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            physics_fn: a PhysicsFN module that provides additional features
            fusion_module: a fusion module that combines ROI features and PhysicsFN features
            roi_align: a ROI Align module that performs ROI pooling
            heads: a ROI head that performs per-region computation and prediction
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        # Initialize Detectron2 base fields (backbone, normalization buffers, etc.)
        # using the same configurable path as GeneralizedRCNN.
        super().__init__()
        self.backbone = backbone

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # self.physics_fn = physics_fn
        # self.fusion_module = fusion_module
        self.roi_align = roi_align
        self.set_transformer = set_transformer
        self.heads = heads

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        output_shapes = backbone.output_shape()
        roi_in_features = cfg.MODEL.ROI_ALIGN.IN_FEATURES
        roi_feature_dim = output_shapes[roi_in_features[0]].channels
        physics_fan_feature_dim = cfg.MODEL.PHYSICS_FAN.EMBEDDING_DIM

        fusion_module = build_fusion_module(
            cfg,
            roi_feature_dim=roi_feature_dim,
            physics_fan_feature_dim=physics_fan_feature_dim,
        )

        physics_fn = build_physics_fn(cfg)

        roi_align = build_roi_align(cfg, output_shapes)

        set_transformer = build_set_transformer(cfg, input_dim=physics_fan_feature_dim + 10)

        # The input dimension for the physics heads is determined by the fusion module's output size
        heads = build_set_heads(
            cfg,
            proposal_input_dim=physics_fan_feature_dim + 10,
            membership_input_dim=physics_fan_feature_dim + 10
        )

        return {
            "backbone": backbone,
            "fusion_module": fusion_module,
            "physics_fn": physics_fn,
            "roi_align": roi_align,
            "set_transformer": set_transformer,
            "heads": heads,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training: # Do inference
            return self.inference(batched_inputs)

        # preprocess the input images and get ground truth
        images = self.preprocess_image(batched_inputs)
        if "target" in batched_inputs[0]:
            gt_targets = [x["target"].to(self.device) for x in batched_inputs]
        else:
            gt_targets = None

        # We assume that proposals are provided in the input for training.
        assert "proposals" in batched_inputs[0]
        proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        # ---- Extract features using the backbone ----
        features = self.backbone(images.tensor)

        # ---- Backbone features -> ROIAling -> features ----
        roi_features = self.roi_align(features, proposals)

        # ---- Extract the locations (dx, dy) of each component in each proposal and pool from ROI features ----
        if "physics_features" in batched_inputs[0]:
            phys_features = [x["physics_features"].to(self.device) for x in batched_inputs]
        else:
            raise NotImplementedError("Physics features are expected in the input for training. Please ensure that the dataset mapper is providing them.")
        (
            dx, # (B, P, C) in [-0.5, 0.5]
            dy, # (B, P, C) in [-0.5, 0.5]
            physics_feats, # (B, P, C, num_physics_features)
            component_mask # (B, P, C) bool tensor indicating valid components
        ) = self._extract_spatial_features(phys_features)
        physics_feats = physics_feats.view(
            -1, physics_feats.shape[-2], physics_feats.shape[-1]
        ) # (B, P, C, num_physics_features) -> (N, C, num_physics_features)

        component_roi_features = self._extract_component_roi_features(
            roi_features, dx, dy, component_mask
        ) # (N, C, D)
        set_input = torch.cat([component_roi_features, physics_feats], dim=-1) # (N, C, D + num_physics_features)

        # ---- Set Transformer ----
        key_padding_mask = ~component_mask.bool().view(-1, component_mask.shape[-1]) # (N, C)
        (
            enc_feats, # (N, C_comp, D)
            dec_feats  # (N, 1, D)
        ) = self.set_transformer(set_input, mask=key_padding_mask) # (N, C_comp, D)

        # ---- Set Transformer features -> Set Heads -> losses ----
        _, detector_losses = self.heads(enc_feats, dec_feats, proposals, gt_targets)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        return losses
    
    def _extract_spatial_features(self, features):
        """
        Extract spatial features (dx, dy) from the input physics features.
        This is a placeholder function and should be implemented based on the actual structure of the input features.
        
        :param features: A tensor of shape (B, P, C, num_physics_features) containing the physics features for each component.
        :return: A tensor of shape (B, P, C, 2) containing the extracted spatial features (dx, dy) for each component.
        """
        # features contains list of Instances with a field component_features of shape
        # (P, C, num_physics_features)
        # We are batching the features, so we need to stack them into a tensor of shape
        # (B, P, C, num_physics_features)
        # The features are already tensors on the right device
        component_features = torch.stack([feat.component_features for feat in features], dim=0)  # (B, P, C, num_physics_features)
        component_mask = torch.stack([feat.component_mask for feat in features], dim=0)  # (B, P, C)

        # Currently we are assuming that the spatial features (dx, dy)
        # are located at specific indices in the physics features.
        dx = component_features[..., 6]  # Assuming the first feature is dx - (B, P, C)
        dy = component_features[..., 7]  # Assuming the second feature is dy - (B, P, C)
        return dx, dy, component_features, component_mask
    
    def _extract_component_roi_features(
            self,
            roi_features,   # (N, D, H, W)
            dx,         # (N, C) in [0, W-1]
            dy,         # (N, C) in [0, H-1]
            component_mask, # (B, P, C)
        ):
        N, D, H, W = roi_features.shape

        # Turn into (B, P, C) -> (N, C) where N=B*P for easier processing
        scaled_dx = dx.reshape(-1, dx.shape[2])  # (N, C)
        scaled_dy = dy.reshape(-1, dy.shape[2])  # (N, C)

        # Shift from [-0.5, 0.5] to [0, 1]
        norm_x = (scaled_dx + 0.5).clamp(0.0, 1.0)  # (N, C)
        norm_y = (scaled_dy + 0.5).clamp(0.0, 1.0)  # (N, C)

        # Map to grid indices [0, W-1] and [0, H-1]
        grid_x = norm_x * (W - 1)  # (N, C)
        grid_y = norm_y * (H - 1)  # (N, C)
        
        # F.grid_sample expects coordinates in [-1, 1]
        # Normalise from [0, W-1] and [0, H-1] to [-1, 1]
        norm_x = (grid_x / (W - 1)) * 2 - 1  # (N, C)
        norm_y = (grid_y / (H - 1)) * 2 - 1  # (N, C)
        
        # grid_sample expects (N, H_out, W_out, 2)
        # We want C output points, shaped as (N, C, 1, 2)
        sample_grid = torch.stack([norm_x, norm_y], dim=-1)  # (N, C, 2)
        sample_grid = sample_grid.unsqueeze(2)               # (N, C, 1, 2)
        
        # Sample from ROI features at component locations
        # Output: (N, D, C, 1)
        sampled = F.grid_sample(
            roi_features,
            sample_grid,
            mode='bilinear',    # bilinear interpolation — smooth gradients
            padding_mode='zeros',
            align_corners=True
        )  # (N, D, C, 1)
        
        component_visual = sampled.squeeze(-1).permute(0, 2, 1)  # (N, C, D)
        
        # Zero out padding components
        component_visual = component_visual * component_mask.reshape(-1, component_mask.shape[2]).unsqueeze(-1).float()
        return component_visual  # (N, C, D)

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            # We assume that proposals are provided in the input for training.
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            # ---- Backbone features -> ROIAling -> features ----
            roi_features = self.roi_align(features, proposals)

            # ---- Extract the locations (dx, dy) of each component in each proposal and pool from ROI features ----
            if "physics_features" in batched_inputs[0]:
                phys_features = [x["physics_features"].to(self.device) for x in batched_inputs]
            else:
                raise NotImplementedError("Physics features are expected in the input for training. Please ensure that the dataset mapper is providing them.")
            (
                dx, # (B, P, C) in [-0.5, 0.5]
                dy, # (B, P, C) in [-0.5, 0.5]
                physics_feats, # (B, P, C, num_physics_features)
                component_mask # (B, P, C) bool tensor indicating valid components
            ) = self._extract_spatial_features(phys_features)
            physics_feats = physics_feats.view(
                -1, physics_feats.shape[-2], physics_feats.shape[-1]
            ) # (B, P, C, num_physics_features) -> (N, C, num_physics_features)

            component_roi_features = self._extract_component_roi_features(
                roi_features, dx, dy, component_mask
            ) # (N, C, D)
            set_input = torch.cat([component_roi_features, physics_feats], dim=-1) # (N, C, D + num_physics_features)

            # ---- Set Transformer ----
            key_padding_mask = ~component_mask.bool().view(-1, component_mask.shape[-1]) # (N, C)
            (
                enc_feats, # (N, C_comp, D)
                dec_feats  # (N, 1, D)
            ) = self.set_transformer(set_input, mask=key_padding_mask) # (N, C_comp, D)

            # ---- Set Transformer features -> Set Heads -> losses ----
            results, _ = self.heads(enc_feats, dec_feats, proposals)
        else:
            results = [x.to(self.device) for x in detected_instances]

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return RaCAST._postprocess(results, batched_inputs)
        return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        processed_results = []
        for r, input_per_image in zip(
            instances, batched_inputs
        ):
            ress = {
                "instances": r
            }

            # If annotations are included in the input, we also include the GT proposal validity
            # and component membership in the output for evaluation purposes.
            # This is not needed for inference, but can be useful for evaluating on a
            # validation set where annotations are available.
            # if 'annotations' in input_per_image:
            #     ress["gt_proposal_validity"] = input_per_image.gt_proposal_validity
            #     ress["gt_component_membership"] = input_per_image.gt_component_membership

            processed_results.append(ress)
        return processed_results
