import os
import torch
import numpy as np
from typing import Dict, Any
from detectron2.data import DatasetMapper
from detectron2.structures import Instances, Boxes


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
            proposal_file = self._get_proposal_file(dataset_dict.get("file_name"))
            
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
    
    def _get_proposal_file(self, image_file):
        """
        Find the NPZ proposal file corresponding to the image.
        
        Args:
            image_file: path to image file (PNG)
            
        Returns:
            str: path to NPZ file, or None if not found
        """
        if image_file is None:
            return None
            
        base_name = os.path.basename(image_file)
        npz_name = os.path.splitext(base_name)[0] + '.npz'
        
        if self.proposal_dir:
            # Use specified proposal directory
            return os.path.join(self.proposal_dir, npz_name)
        else:
            # Auto-detect: try multiple common locations
            image_dir = os.path.dirname(image_file)
            candidate = os.path.join(image_dir.replace('images', 'proposals'), npz_name)

            if os.path.exists(candidate):
                return candidate
            
            return None