import os
import sys
import logging
from pathlib import Path

# Add detectron2 to path (assumes detectron2 is in the parent directory structure)
detectron2_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(detectron2_root))

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.logger import create_small_table

import numpy as np
import torch

from .probe import COCOProbe


class GRGEvaluator(DatasetEvaluator, COCOProbe):
    """
    If our predicted region uniquely encompasses the central
    coordinates of the (non-removed or reinserted) radiocomponents
    in accordance with the manual association, we have a true positive (TP).
    If the region does not encompass all of
    the radio components that belong together, we have a false positive (FP).
    If the region encompasses all the radio components that belong together,
    but also encompasses additional unrelated radio components, that also counts as a FP.
    If there is no region covering the central coordinate of the focussed radio component
    with a score surpassing the user-set threshold we have a false
    negative (FN). A true negative (TN) is the absence of a region
    where this is indeed warranted. True negatives should not appear
    in our data, as we only consider radio images centred on radio
    components with a signal-to-noise ratio surpassing five.
    """
    def __init__(self, coco_images: list[dict], annotations_path: str, score_threshold: float = 0.5):
        super().__init__(annotations_path)
        self.coco_images = coco_images
        self._cpu_device = torch.device("cpu")
        self._score_threshold = score_threshold
        self._logger = logging.getLogger(__name__)

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input_data, output in zip(inputs, outputs):
            prediction = {
                "image_id": input_data["image_id"],
                "instances": output["instances"].to(self._cpu_device) if "instances" in output else None
            }
            self._predictions.append(prediction)

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        if len(self._predictions) == 0:
            self._logger.warning("[GRGEvaluator] Did not receive valid predictions.")
            return {}
        
        tp_mask, fp_mask, fn_mask, tp_bbox, fp_bbox, fn_bbox = self._gather_predictions()
        
        # Compute metrics
        results = {
            "segm_accuracy": self._accuracy(tp_mask, fp_mask, fn_mask),
            "segm_precision": self._precision(tp_mask, fp_mask),
            "segm_recall": self._recall(tp_mask, fn_mask),
            "bbox_accuracy": self._accuracy(tp_bbox, fp_bbox, fn_bbox),
            "bbox_precision": self._precision(tp_bbox, fp_bbox),
            "bbox_recall": self._recall(tp_bbox, fn_bbox)
        }
        
        # Log the results in a nice table format
        self._logger.info("GRG Evaluation Results:\n" + create_small_table(results))
        return {
            "GRG": results
        }

    def _gather_predictions(self):
        tp_mask_list = []
        fp_mask_list = []
        fn_mask_list = []
        tp_bbox_list = []
        fp_bbox_list = []
        fn_bbox_list = []
        
        # Create a mapping from image_id to image metadata
        # First load the image by id, then get the metadata from the coco annotations
        image_id_to_metadata = {
            img['image_id']: self.coco.loadImgs(img['image_id'])[0]['metadata']
            for img in self.coco_images
        }
        
        for prediction in self._predictions:
            image_id = prediction['image_id']
            image_metadata = image_id_to_metadata.get(image_id)
            
            # Log a warning if the image_id from predictions is not found in the coco annotations
            # This should not happen but better safe than sorry
            if image_metadata is None:
                self._logger.warning(f"Image ID {image_id} not found in annotations")
                continue
            
            # Get the predicted mask and bounding box for this image
            valid_instances = self._get_valid_instances(prediction)
            mask = self._get_mask_from_instances(valid_instances)
            bbox = self._get_bbox_from_instances(valid_instances)

            # Extract the GRG components and non-GRG components for this image
            grg_components = self._extract_gt_components(image_metadata)
            all_components = self._extract_all_components(image_metadata)
            non_grg_components = self._remove_grg_from_all_components(all_components, grg_components)

            # Check if the GRG components are in the predicted mask and bounding box
            grg_components_in_mask = self._grg_components_are_in_mask(grg_components, mask)
            grg_components_in_bbox = self._grg_components_are_in_bbox(grg_components, bbox)
    
            # Check if the non-GRG components are in the predicted mask
            non_grg_components_in_mask = self._non_grg_components_are_in_mask(non_grg_components, mask)

            # Calculate TP, FP, FN for both mask and bbox evaluations
            tp_mask = self._tp(grg_components_in_mask, non_grg_components_in_mask)
            fp_mask = self._fp(grg_components_in_mask, non_grg_components_in_mask)
            fn_mask = self._fn(grg_components_in_mask, non_grg_components_in_mask)
            tp_bbox = self._tp(grg_components_in_bbox, non_grg_components_in_mask)
            fp_bbox = self._fp(grg_components_in_bbox, non_grg_components_in_mask)
            fn_bbox = self._fn(grg_components_in_bbox, non_grg_components_in_mask)

            # Append results to lists for later aggregation
            tp_mask_list.append(tp_mask)
            fp_mask_list.append(fp_mask)
            fn_mask_list.append(fn_mask)
            tp_bbox_list.append(tp_bbox)
            fp_bbox_list.append(fp_bbox)
            fn_bbox_list.append(fn_bbox)

        # Convert to numpy arrays for easier calculation of metrics
        # and also convert the bool values to integers (1 for True, 0 for False)
        # for metric calculations
        tp_mask_list = np.array(tp_mask_list).astype(int)
        fp_mask_list = np.array(fp_mask_list).astype(int)
        fn_mask_list = np.array(fn_mask_list).astype(int)
        tp_bbox_list = np.array(tp_bbox_list).astype(int)
        fp_bbox_list = np.array(fp_bbox_list).astype(int)
        fn_bbox_list = np.array(fn_bbox_list).astype(int)
        
        return tp_mask_list, fp_mask_list, fn_mask_list, tp_bbox_list, fp_bbox_list, fn_bbox_list
    
    def _get_valid_instances(self, prediction):
        """
        Extract and filter instances from predictions.
        Filters by score threshold and keeps only the highest-confidence prediction.
        
        Args:
            prediction (dict): A dict containing 'instances' with detectron2 Instances object
            
        Returns:
            Instances object or None: Filtered instances (max 1) or None if no valid predictions
        """
        instances = prediction.get('instances')
        
        if instances is None or len(instances) == 0:
            return None
        
        # Filter instances by score threshold
        scores = instances.scores
        valid_indices = scores >= self._score_threshold
        
        if valid_indices.sum() == 0:
            return None
        
        # Get valid instances
        valid_instances = instances[valid_indices]
        
        # Keep only the highest-confidence prediction
        if len(valid_instances) > 1:
            best_idx = valid_instances.scores.argmax()
            valid_instances = valid_instances[best_idx:best_idx+1]
        
        return valid_instances
    
    def _get_bbox_from_instances(self, valid_instances):
        """
        Extract bounding box from valid instances.
        
        Args:
            valid_instances: Filtered Instances object or None
            
        Returns:
            list: A list of bounding boxes in the format [x1, y1, x2, y2]
        """
        if valid_instances is None:
            return []
        
        # Extract bounding boxes
        bboxes = []
        if hasattr(valid_instances, 'pred_boxes'):
            # pred_boxes.tensor gives us the tensor, take first (and only) box
            box = valid_instances.pred_boxes.tensor[0].int().numpy()
            bboxes.append(box)
        
        return bboxes
    
    def _get_mask_from_instances(self, valid_instances):
        """
        Convert valid instances to a binary mask that can be used for evaluation.
        
        Args:
            valid_instances: Filtered Instances object or None
            
        Returns:
            np.ndarray: Binary mask where 1 indicates predicted region, 0 background
        """
        if valid_instances is None:
            # No valid instances - return empty mask with default dimensions
            return np.zeros((300, 300), dtype=np.uint8)
        
        # Get image dimensions from the instances object
        height = valid_instances.image_size[0]
        width = valid_instances.image_size[1]
        
        # Initialize combined mask
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Combine all masks
        if hasattr(valid_instances, 'pred_masks'):
            # Instance segmentation: combine all predicted masks
            # for mask in valid_instances.pred_masks:
            #     combined_mask = np.logical_or(combined_mask, mask.numpy())
            mask = valid_instances.pred_masks[0].numpy()
            combined_mask = np.logical_or(combined_mask, mask)
        elif hasattr(valid_instances, 'pred_boxes'):
            # Detection only: use bounding boxes as masks
            for box in valid_instances.pred_boxes:
                x1, y1, x2, y2 = box.tensor[0].int().numpy()
                combined_mask[y1:y2, x1:x2] = 1
        
        return combined_mask.astype(np.uint8)
    
    def _grg_components_are_in_mask(self, grg_components: list, mask: np.ndarray):
        """
        Check if the given components (list of tuples) are within the predicted mask (2D numpy array).
        """
        for comp in grg_components:
            x, y = comp
            # Assuming mask is binary with 1 for predicted region and 0 for background
            if mask[int(y), int(x)] == 0:
                return False
        return True
    
    def _grg_components_are_in_bbox(self, grg_components: list, bbox: list):
        """
        Check if the given components (list of tuples) are within the predicted mask (2D numpy array).
        """
        for comp in grg_components:
            x, y = comp
            if bbox:
                # Assuming bbox is a list of one bounding box
                x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]
                if (x1 <= x <= x2 and y1 <= y <= y2):
                    # If the component is within the bounding box,
                    # we can consider it as covered by the prediction,
                    # even if it's not in the mask (for detection-only models)
                    continue
                return False
        return True
    
    def _non_grg_components_are_in_mask(self, non_grg_components: list, mask: np.ndarray):
        """
        Check if the given components (list of tuples) are within the predicted mask (2D numpy array).
        """
        for comp in non_grg_components:
            x, y = comp
            # Assuming mask is binary with 1 for predicted region and 0 for background
            if mask[int(y), int(x)] == 1:
                return True
        return False
    
    def _accuracy(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray):
        """Calculate accuracy from TP, FP, FN"""
        total = np.sum(tp) + np.sum(fp) + np.sum(fn)
        correct = np.sum(tp)
        return correct / total if total > 0 else 0.0

    def _precision(self, tp: np.ndarray, fp: np.ndarray):
        """Calculate precision from TP and FP"""
        tp_sum = np.sum(tp)
        fp_sum = np.sum(fp)
        return tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        
    def _recall(self, tp: np.ndarray, fn: np.ndarray):
        """Calculate recall from TP and FN"""
        tp_sum = np.sum(tp)
        fn_sum = np.sum(fn)
        return tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0

    def _tp(self, grg_in_mask_or_bbox: bool, non_grg_in_mask: bool): # True Positives
        """Region uniquely encompasses all GRG components and no non-GRG components"""
        if grg_in_mask_or_bbox == True and non_grg_in_mask == False:
            return True
        return False

    def _fp(self, grg_in_mask_or_bbox: bool, non_grg_in_mask: bool): # False Positives
        """Region missing GRG components OR includes non-GRG components"""
        if grg_in_mask_or_bbox == False or non_grg_in_mask == True:
            return True
        return False

    def _fn(self, grg_in_mask_or_bbox: bool, non_grg_in_mask: bool): # False Negatives
        """No region covering GRG components (should be: not grg_in_mask_or_bbox)"""
        if grg_in_mask_or_bbox == False:
            return True
        return False
