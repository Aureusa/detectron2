import os
import torch
import numpy as np
from typing import Dict, Any
from detectron2.data import DatasetMapper
from detectron2.structures import Instances, Boxes
import copy

import data.utils as utils
from structures import Membership, Validity


class GRGDatasetMapper(DatasetMapper):
    """
    Custom mapper that loads precomputed proposals from individual NPZ files.
    Each image has a corresponding NPZ file with 'boxes' and 'scores' arrays.
    """
    
    def __init__(self, cfg, is_train=True, proposal_dir=None):
        """
        Args:
            cfg: Detectron2 config
            is_train: whether in training mode
            proposal_dir: directory containing NPZ proposal files.
                         If None, will auto-detect based on image path
        """
        super().__init__(cfg, is_train)
        self.proposal_dir = proposal_dir
        
        # Set proposal_topk from config if not already set by parent
        # This ensures we load proposals even when MODEL.LOAD_PROPOSALS is False
        if self.proposal_topk is None:
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        
    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load image, annotations, and precomputed proposals from NPZ file.
        
        Args:
            dataset_dict: dict with "file_name" and other standard fields
        
        Returns:
            dict with added "proposals" field if proposals are enabled
        """
        # Call parent mapper to handle standard processing
        dataset_dict = super().__call__(dataset_dict)
        
        # Only load proposals if enabled in config
        if self.proposal_topk is not None:
            proposal_file = utils.get_proposal_file(dataset_dict.get("file_name"), self.proposal_dir)
            
            if proposal_file:
                try:
                    # Load NPZ file
                    npz_data = np.load(proposal_file)
                    
                    # Extract boxes and scores
                    # Your PrecomputeProposals.precompute() returns (boxes, scores)
                    boxes = npz_data['boxes']  # Shape: (N, 4), format: [x1, y1, x2, y2]
                    scores = npz_data['scores']  # Shape: (N,), normalized [0, 1]
                    
                    # Get image dimensions from the transformed image
                    h, w = dataset_dict["image"].shape[1:]  # CHW format
                    
                    # Convert to Detectron2 Instances
                    proposals = Instances((h, w))
                    proposals.proposal_boxes = Boxes(torch.from_numpy(boxes).float())
                    proposals.objectness_logits = torch.from_numpy(scores).float()
                    
                    # Keep only top-k proposals
                    if len(proposals) > self.proposal_topk:
                        # Sort by objectness scores (descending) and keep top-k
                        _, indices = torch.topk(
                            proposals.objectness_logits, 
                            min(self.proposal_topk, len(proposals))
                        )
                        proposals = proposals[indices]
                    
                    dataset_dict["proposals"] = proposals
                    
                except Exception as e:
                    # Log error but don't crash - training can continue without proposals
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to load proposals from {proposal_file}: {e}")
        
        return dataset_dict
    

class B2SDatasetMapper(DatasetMapper):
    """
    Custom mapper that loads precomputed proposals from individual NPZ files.
    Each image has a corresponding NPZ file with 'boxes' and 'scores' arrays.
    """
    
    def __init__(self, cfg, is_train=True, proposal_dir=None):
        """
        Args:
            cfg: Detectron2 config
            is_train: whether in training mode
            proposal_dir: directory containing NPZ proposal files.
                         If None, will auto-detect based on image path
        """
        super().__init__(cfg, is_train)
        self.proposal_dir = proposal_dir
        
        # Set proposal_topk from config if not already set by parent
        # This ensures we load proposals even when MODEL.LOAD_PROPOSALS is False
        if self.proposal_topk is None:
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        
    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load image, annotations, and precomputed proposals from NPZ file.
        
        Args:
            dataset_dict: dict with "file_name" and other standard fields
        
        Returns:
            dict with added "proposals" field if proposals are enabled
        """
        # Preserve raw annotations payload for custom per-proposal GT fields
        raw_annotations = copy.deepcopy(dataset_dict.get("annotations", []))

        # Load the image
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        proposal_file = utils.get_proposal_file(dataset_dict.get("file_name"), self.proposal_dir)

        if proposal_file:
            try:
                # Load NPZ file
                npz_data = np.load(proposal_file)

                # Extract boxes, physical features and masks
                boxes = npz_data['boxes']  # Shape: (N, 4), format: [x1, y1, x2, y2]
                features = npz_data['features']  # Shape: (N, C, F) features per proposal-component
                masks = npz_data['within_proposal_mask']  # Shape: (N, C), masks for each proposal-component
                scores = npz_data['scores'] if 'scores' in npz_data else np.ones((boxes.shape[0],), dtype=np.float32)

                # Get image dimensions from the transformed image
                h, w = dataset_dict["height"], dataset_dict["width"]

                # Convert proposals to Detectron2 Instances
                proposals = Instances((h, w))
                proposals.proposal_boxes = Boxes(torch.from_numpy(boxes).float())
                proposals.objectness_logits = torch.from_numpy(scores).float()

                # Keep component data separate from proposals as requested
                physical_features = Instances((h, w))
                physical_features.component_mask = torch.from_numpy(masks).bool()
                physical_features.component_features = torch.from_numpy(features).float()

                # Build a separate target Instances for proposal-level supervision
                target = self._build_membership_validity_target((h, w), masks, raw_annotations)

                dataset_dict["proposals"] = proposals
                dataset_dict["physics_features"] = physical_features
                dataset_dict["target"] = target

            except Exception as e:
                # Log error but don't crash - training can continue without proposals
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load proposals from {proposal_file}: {e}")
        
        return dataset_dict

    def _build_membership_validity_target(self, image_size, masks: np.ndarray, annotations: Any) -> Instances:
        target = Instances(image_size)
        target.component_mask = torch.from_numpy(masks).bool()

        if not annotations:
            return target

        payload = None
        for anno in annotations:
            if "gt_component_membership" in anno and "gt_proposal_validity" in anno:
                payload = anno
                break
        if payload is None:
            return target

        membership = torch.as_tensor(payload["gt_component_membership"], dtype=torch.float32)
        validity = torch.as_tensor(payload["gt_proposal_validity"], dtype=torch.float32)

        num_props = masks.shape[0]
        num_components = masks.shape[1]

        # Ensure [P, C] and [P] shapes aligned to proposal count
        if membership.dim() != 2:
            raise ValueError(f"gt_component_membership must be 2D, got {tuple(membership.shape)}")

        if membership.shape[1] < num_components:
            pad = torch.zeros((membership.shape[0], num_components - membership.shape[1]), dtype=membership.dtype)
            membership = torch.cat([membership, pad], dim=1)
        membership = membership[:, :num_components]

        if membership.shape[0] < num_props:
            pad = torch.zeros((num_props - membership.shape[0], membership.shape[1]), dtype=membership.dtype)
            membership = torch.cat([membership, pad], dim=0)
        membership = membership[:num_props]

        if validity.dim() != 1:
            validity = validity.reshape(-1)
        if validity.shape[0] < num_props:
            pad = torch.zeros((num_props - validity.shape[0],), dtype=validity.dtype)
            validity = torch.cat([validity, pad], dim=0)
        validity = validity[:num_props]

        target.gt_component_membership = Membership(membership)
        target.gt_proposal_validity = Validity(validity)
        return target
