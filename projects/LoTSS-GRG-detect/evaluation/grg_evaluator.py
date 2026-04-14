import sys
import logging
from pathlib import Path
import copy

# Add detectron2 to path (assumes detectron2 is in the parent directory structure)
detectron2_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(detectron2_root))

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.logger import create_small_table, setup_logger

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
        self._logger = setup_logger(name="LoTSS-GRG-detect.evaluation.GRGEvaluator", termcolor="magenta")

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

        segm_precision = self._precision(tp_mask, fp_mask)
        segm_recall = self._recall(tp_mask, fn_mask)

        bbox_precision = self._precision(tp_bbox, fp_bbox)
        bbox_recall = self._recall(tp_bbox, fn_bbox)
        
        # Compute metrics
        results = {
            "segm_accuracy": self._accuracy(tp_mask, fp_mask, fn_mask),
            "segm_precision": segm_precision,
            "segm_recall": segm_recall,
            "segm_f1": self._f1(segm_precision, segm_recall),
            "bbox_accuracy": self._accuracy(tp_bbox, fp_bbox, fn_bbox),
            "bbox_precision": bbox_precision,
            "bbox_recall": bbox_recall,
            "bbox_f1": self._f1(bbox_precision, bbox_recall)
        }

        segm_results = {
            "Segmentation Accuracy": results["segm_accuracy"],
            "Segmentation Precision": results["segm_precision"],
            "Segmentation Recall": results["segm_recall"],
            "Segmentation F1": results["segm_f1"]
        }

        bbox_results = {
            "Bbox Accuracy": results["bbox_accuracy"],
            "Bbox Precision": results["bbox_precision"],
            "Bbox Recall": results["bbox_recall"],
            "Bbox F1": results["bbox_f1"]
        }
        
        # Log the results in a nice table format
        self._logger.info("GRG Evaluation Results:\n" + create_small_table(segm_results) + "\n" + create_small_table(bbox_results))
        return copy.deepcopy({
            "GRG": results
        })

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
                self._logger.warning(f"Image ID {image_id} not found in COCO annotations. Skipping this prediction.")
                continue

            if not image_metadata.get("grg_in_sample", True):
                # For samples where there is no known grg (refer to as negative samples),
                # we expect no predictions.
                # If there are predictions, they are false positives.
                has_prediction = prediction['instances'] is not None and len(prediction['instances']) > 0
                tp_mask_list.append(0)  # No true positives in negative samples
                fp_mask_list.append(1 if has_prediction else 0)  # False positive if there is any prediction
                fn_mask_list.append(0)  # No false negatives in negative samples
                tp_bbox_list.append(0)  # No true positives in negative samples
                fp_bbox_list.append(1 if has_prediction else 0)  # False positive if there is any prediction
                fn_bbox_list.append(0)  # No false negatives in negative samples
                continue
            
            # Get the predicted mask and bounding box for this image
            valid_instances = self._get_valid_instances(prediction)
            mask = self._get_mask_from_instances(valid_instances)
            bbox = self._get_bbox_from_instances(valid_instances)

            # Extract the GRG components and non-GRG components for this image
            grg_components = self._extract_gt_components(image_metadata)
            all_components = self._extract_all_components(image_metadata)
            
            # Convert grg_components and all_components from dict to list if they are dicts
            if isinstance(grg_components, dict):
                grg_components = list(grg_components.values())
            if isinstance(all_components, dict):
                all_components = list(all_components.values())

            non_grg_components = self._remove_grg_from_all_components(all_components, grg_components)

            # Check if the GRG components are in the predicted mask and bounding box
            all_grg_components_in_mask, some_grg_components_in_mask = self._grg_components_are_in_mask(
                grg_components, mask
            )
            all_grg_components_in_bbox, some_grg_components_in_bbox = self._grg_components_are_in_bbox(
                grg_components, bbox
            )
    
            # Check if the non-GRG components are in the predicted mask
            non_grg_components_in_mask = self._non_grg_components_are_in_mask(non_grg_components, mask)

            # Calculate TP, FP, FN for both mask and bbox evaluations
            tp_mask = self._tp(all_grg_components_in_mask, non_grg_components_in_mask)
            fp_mask = self._fp(all_grg_components_in_mask, some_grg_components_in_mask, non_grg_components_in_mask)
            fn_mask = self._fn(some_grg_components_in_mask)
            tp_bbox = self._tp(all_grg_components_in_bbox, non_grg_components_in_mask)
            fp_bbox = self._fp(all_grg_components_in_bbox, some_grg_components_in_bbox, non_grg_components_in_mask)
            fn_bbox = self._fn(some_grg_components_in_bbox)

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
        all_in_mask = None
        some_in_mask = None
        for comp in grg_components:
            x, y = comp
            # Assuming mask is binary with 1 for predicted region and 0 for background
            if mask[int(y), int(x)] == 0:
                all_in_mask = False
                continue
            some_in_mask = True

        # If we never set some_in_mask to True,
        # it means none of the components are in the mask,
        # so we set it to False
        # If we never set all_in_mask to False,
        # it means all components are in the mask,
        # so we set it to True
        if all_in_mask == False and some_in_mask == None:
            some_in_mask = False
        if some_in_mask == True and all_in_mask == None:
            all_in_mask = True
        return all_in_mask, some_in_mask
    
    def _grg_components_are_in_bbox(self, grg_components: list, bbox: list):
        """
        Check if the given components (list of tuples) are within the predicted mask (2D numpy array).
        """
        if not bbox:  # No bbox at all
            return False, False
    
        all_in_bbox = None
        some_in_bbox = None
        for comp in grg_components:
            x, y = comp
            if bbox:
                # Assuming bbox is a list of one bounding box
                x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]
                if (x1 <= x <= x2 and y1 <= y <= y2):
                    # If the component is within the bounding box,
                    # we can consider it as covered by the prediction,
                    # even if it's not in the mask (for detection-only models)
                    some_in_bbox = True
                    continue
                all_in_bbox = False

        # If we never set some_in_bbox to True,
        # it means none of the components are in the bounding box,
        # so we set it to False
        # If we never set all_in_bbox to False,
        # it means all components are in the bounding box,
        # so we set it to True
        if all_in_bbox == False and some_in_bbox == None:
            some_in_bbox = False
        if some_in_bbox == True and all_in_bbox == None:
            all_in_bbox = True
        return all_in_bbox, some_in_bbox
    
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
    
    def _f1(self, precision: float, recall: float):
        """Calculate F1 score from precision and recall"""
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def _tp(self, all_components_in: bool, non_grg_in_mask: bool):
        """Region uniquely encompasses all GRG components and no non-GRG components"""
        return all_components_in == True and non_grg_in_mask == False

    def _fp(self, all_components_in: bool, some_components_in: bool, non_grg_in_mask: bool):
        """Made a prediction that's wrong (partial or includes extras)"""
        # If no prediction at all, it's FN not FP
        if some_components_in == False:
            return False
        # If we detected something but it's imperfect
        return all_components_in == False or non_grg_in_mask == True

    def _fn(self, some_components_in: bool):
        """Failed to detect GRG components"""
        return some_components_in == False


class B2SMaskedRCNNEvaluator(GRGEvaluator):
    """
    Evaluator for Mask R-CNN model trained with B2S approach.
    Inherits from GRGEvaluator and can override methods if needed for specific handling of Mask R-CNN outputs.
    """
    def __init__(self, coco_images: list[dict], annotations_path: str, score_threshold: float = 0.5):
        super().__init__(coco_images, annotations_path, score_threshold)
        self._category_id_to_name = self._build_category_name_mapping()
        self._dataset_id_to_contiguous_id = self._build_category_id_mapping()

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
                "instances": output["instances"].to(self._cpu_device) if "instances" in output else None,
            }
            self._predictions.append(prediction)

    def evaluate(self):
        """
        Evaluate aggregate and class-wise metrics using annotation instance_positions.
        """
        if len(self._predictions) == 0:
            self._logger.warning("[B2SMaskedRCNNEvaluator] Did not receive valid predictions.")
            return {}

        overall, classwise = self._gather_predictions_with_classwise_counts()

        segm_precision = self._precision(overall["mask"]["tp"], overall["mask"]["fp"])
        segm_recall = self._recall(overall["mask"]["tp"], overall["mask"]["fn"])
        bbox_precision = self._precision(overall["bbox"]["tp"], overall["bbox"]["fp"])
        bbox_recall = self._recall(overall["bbox"]["tp"], overall["bbox"]["fn"])

        results = {
            "segm_accuracy": self._accuracy(overall["mask"]["tp"], overall["mask"]["fp"], overall["mask"]["fn"]),
            "segm_precision": segm_precision,
            "segm_recall": segm_recall,
            "segm_f1": self._f1(segm_precision, segm_recall),
            "bbox_accuracy": self._accuracy(overall["bbox"]["tp"], overall["bbox"]["fp"], overall["bbox"]["fn"]),
            "bbox_precision": bbox_precision,
            "bbox_recall": bbox_recall,
            "bbox_f1": self._f1(bbox_precision, bbox_recall),
        }

        class_results = {}
        for class_id, counts in sorted(classwise.items(), key=lambda item: item[0]):
            class_name = self._class_id_to_name(class_id)

            segm_p = self._precision(counts["mask"]["tp"], counts["mask"]["fp"])
            segm_r = self._recall(counts["mask"]["tp"], counts["mask"]["fn"])
            bbox_p = self._precision(counts["bbox"]["tp"], counts["bbox"]["fp"])
            bbox_r = self._recall(counts["bbox"]["tp"], counts["bbox"]["fn"])

            class_results[class_name] = {
                "segm_precision": segm_p,
                "segm_recall": segm_r,
                "segm_f1": self._f1(segm_p, segm_r),
                "bbox_precision": bbox_p,
                "bbox_recall": bbox_r,
                "bbox_f1": self._f1(bbox_p, bbox_r),
                "mask_tp": counts["mask"]["tp"],
                "mask_fp": counts["mask"]["fp"],
                "mask_fn": counts["mask"]["fn"],
                "bbox_tp": counts["bbox"]["tp"],
                "bbox_fp": counts["bbox"]["fp"],
                "bbox_fn": counts["bbox"]["fn"],
            }

        overall_table = {
            "Segm Precision": results["segm_precision"],
            "Segm Recall": results["segm_recall"],
            "Segm F1": results["segm_f1"],
            "Bbox Precision": results["bbox_precision"],
            "Bbox Recall": results["bbox_recall"],
            "Bbox F1": results["bbox_f1"],
        }
        self._logger.info("B2S Masked R-CNN (overall):\n" + create_small_table(overall_table))

        for class_name, metrics in class_results.items():
            class_table = {
                "Segm Precision": metrics["segm_precision"],
                "Segm Recall": metrics["segm_recall"],
                "Segm F1": metrics["segm_f1"],
                "Bbox Precision": metrics["bbox_precision"],
                "Bbox Recall": metrics["bbox_recall"],
                "Bbox F1": metrics["bbox_f1"],
                "Mask TP": metrics["mask_tp"],
                "Mask FP": metrics["mask_fp"],
                "Mask FN": metrics["mask_fn"],
                "Bbox TP": metrics["bbox_tp"],
                "Bbox FP": metrics["bbox_fp"],
                "Bbox FN": metrics["bbox_fn"],
            }
            self._logger.info(f"B2S Masked R-CNN ({class_name}):\n" + create_small_table(class_table))

        flat_class_results = {}
        for class_name, metrics in class_results.items():
            for metric_name, metric_value in metrics.items():
                flat_class_results[f"{class_name}_{metric_name}"] = metric_value

        return copy.deepcopy({
            "B2S": results,
            "B2S_CLASSWISE": flat_class_results,
        })
    
    def set_score_threshold(self, threshold: float):
        """Set the score threshold for filtering predictions."""
        self._score_threshold = threshold

    def _gather_predictions(self):
        tp_mask_list = []
        fp_mask_list = []
        fn_mask_list = []
        tp_bbox_list = []
        fp_bbox_list = []
        fn_bbox_list = []
        
        for prediction in self._predictions:
            gt_instances = prediction.get("gt_instances", [])
            pred_instances = self._get_all_valid_instances(prediction)

            tp_mask, fp_mask, fn_mask = self._compute_image_counts(
                gt_instances, pred_instances, use_mask=True
            )
            tp_bbox, fp_bbox, fn_bbox = self._compute_image_counts(
                gt_instances, pred_instances, use_mask=False
            )

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

    def _gather_predictions_with_classwise_counts(self):
        """
        Gather overall and per-class TP/FP/FN counts for mask and bbox containment.
        """
        overall = {
            "mask": {"tp": 0, "fp": 0, "fn": 0},
            "bbox": {"tp": 0, "fp": 0, "fn": 0},
        }
        classwise = {}

        for prediction in self._predictions:
            gt_instances, all_positions = self._extract_gt_instances_from_coco(prediction["image_id"])
            pred_instances = self._get_all_valid_instances(prediction)

            mask_counts, mask_class_counts = self._compute_image_counts_with_classes(
                gt_instances, pred_instances, all_positions, use_mask=True
            )
            bbox_counts, bbox_class_counts = self._compute_image_counts_with_classes(
                gt_instances, pred_instances, all_positions, use_mask=False
            )

            for key in ("tp", "fp", "fn"):
                overall["mask"][key] += mask_counts[key]
                overall["bbox"][key] += bbox_counts[key]

            self._merge_class_counts(classwise, mask_class_counts, eval_type="mask")
            self._merge_class_counts(classwise, bbox_class_counts, eval_type="bbox")

        for class_id in self._known_class_ids():
            if class_id not in classwise:
                classwise[class_id] = {
                    "mask": {"tp": 0, "fp": 0, "fn": 0},
                    "bbox": {"tp": 0, "fp": 0, "fn": 0},
                }

        return overall, classwise

    def _extract_gt_instances_from_coco(self, image_id):
        """Load GT instance positions from the COCO annotations for one image."""
        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        return self._extract_gt_instances(anns)

    def _build_category_name_mapping(self):
        category_id_to_name = {}
        for cat_id in sorted(self.coco.getCatIds()):
            category = self.coco.cats.get(cat_id, {})
            category_id_to_name[cat_id] = category.get("name", f"class_{cat_id}")
        return category_id_to_name

    def _build_category_id_mapping(self):
        """Map dataset category ids to Detectron2 contiguous ids."""
        category_ids = sorted(self.coco.getCatIds())
        return {cat_id: idx for idx, cat_id in enumerate(category_ids)}

    def _known_class_ids(self):
        return sorted(self._dataset_id_to_contiguous_id.values())

    def _extract_gt_instances(self, annotations):
        """Extract per-annotation GT payload used for matching against predictions."""
        gt_instances = []
        if not annotations:
            return gt_instances, set()
        
        all_positions = set()
        for ann in annotations:
            positions = ann.get("instance_positions", [])
            category_id = int(ann.get("category_id", -1))

            normalized_positions = []
            for pos in positions:
                if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                    continue
                x, y = int(pos[0]), int(pos[1])
                normalized_positions.append((x, y))

            if category_id < 0 or len(normalized_positions) == 0:
                continue

            gt_instances.append(
                {
                    "category_id": category_id - 1,  # Convert to 0-based index
                    "positions": normalized_positions,
                }
            )

            all_positions.update(normalized_positions)

        return gt_instances, all_positions

    def _get_all_valid_instances(self, prediction):
        """Keep all predictions above threshold (do not collapse to top-1)."""
        instances = prediction.get("instances")
        if instances is None or len(instances) == 0:
            return None

        scores = instances.scores
        valid_indices = scores >= self._score_threshold
        if valid_indices.sum() == 0:
            return None
        return instances[valid_indices]

    def _compute_image_counts(self, gt_instances, pred_instances, use_mask=True):
        """
        Match GT and predictions one-to-one using class-aware containment.
        TP: matched GT, FN: unmatched GT, FP: unmatched predictions.
        """
        num_gt = len(gt_instances)
        num_pred = 0 if pred_instances is None else len(pred_instances)

        if num_gt == 0:
            return 0, num_pred, 0
        if num_pred == 0:
            return 0, 0, num_gt

        matched_gt = set()
        matched_pred = set()

        for gt_idx, gt in enumerate(gt_instances):
            best_pred_idx = None
            best_score = -1.0

            for pred_idx in range(num_pred):
                if pred_idx in matched_pred:
                    continue

                pred = pred_instances[pred_idx]
                pred_class = int(pred.pred_classes.item()) if hasattr(pred, "pred_classes") else -1
                if pred_class != gt["category_id"]:
                    continue

                if use_mask:
                    contains = self._positions_in_pred_mask(gt["positions"], pred)
                else:
                    contains = self._positions_in_pred_bbox(gt["positions"], pred)

                if not contains:
                    continue

                pred_score = float(pred.scores.item()) if hasattr(pred, "scores") else 0.0
                if pred_score > best_score:
                    best_score = pred_score
                    best_pred_idx = pred_idx

            if best_pred_idx is not None:
                matched_gt.add(gt_idx)
                matched_pred.add(best_pred_idx)

        tp = len(matched_gt)
        fn = num_gt - tp
        fp = num_pred - len(matched_pred)
        return tp, fp, fn

    def _compute_image_counts_with_classes(self, gt_instances, pred_instances, all_positions, use_mask=True):
        """Compute per-image counts and keep per-class TP/FP/FN accounting."""
        num_gt = len(gt_instances)
        num_pred = 0 if pred_instances is None else len(pred_instances)

        class_counts = {}
        if num_gt == 0:
            if num_pred > 0:
                for pred_idx in range(num_pred):
                    pred = pred_instances[pred_idx]
                    pred_class = int(pred.pred_classes.item()) if hasattr(pred, "pred_classes") else -1
                    self._inc_class_count(class_counts, pred_class, "fp")
            return {"tp": 0, "fp": num_pred, "fn": 0}, class_counts

        if num_pred == 0:
            for gt in gt_instances:
                self._inc_class_count(class_counts, gt["category_id"], "fn")
            return {"tp": 0, "fp": 0, "fn": num_gt}, class_counts

        matched_gt = set()
        matched_pred = set()

        for gt_idx, gt in enumerate(gt_instances):
            best_pred_idx = None
            best_score = -1.0

            for pred_idx in range(num_pred):
                if pred_idx in matched_pred:
                    continue

                pred = pred_instances[pred_idx]
                pred_class = int(pred.pred_classes.item()) if hasattr(pred, "pred_classes") else -1
                if pred_class != gt["category_id"]:
                    continue

                if use_mask:
                    contains = self._positions_in_pred_mask(gt["positions"], pred, all_positions)
                else:
                    contains = self._positions_in_pred_bbox(gt["positions"], pred, all_positions)

                if not contains:
                    continue

                pred_score = float(pred.scores.item()) if hasattr(pred, "scores") else 0.0
                if pred_score > best_score:
                    best_score = pred_score
                    best_pred_idx = pred_idx

            if best_pred_idx is not None:
                matched_gt.add(gt_idx)
                matched_pred.add(best_pred_idx)

        for gt_idx, gt in enumerate(gt_instances):
            if gt_idx in matched_gt:
                self._inc_class_count(class_counts, gt["category_id"], "tp")
            else:
                self._inc_class_count(class_counts, gt["category_id"], "fn")

        for pred_idx in range(num_pred):
            if pred_idx in matched_pred:
                continue
            pred = pred_instances[pred_idx]
            pred_class = int(pred.pred_classes.item()) if hasattr(pred, "pred_classes") else -1
            self._inc_class_count(class_counts, pred_class, "fp")

        tp = len(matched_gt)
        fn = num_gt - tp
        fp = num_pred - len(matched_pred)
        return {"tp": tp, "fp": fp, "fn": fn}, class_counts

    def _merge_class_counts(self, aggregate, new_counts, eval_type):
        for class_id, counts in new_counts.items():
            if class_id not in aggregate:
                aggregate[class_id] = {
                    "mask": {"tp": 0, "fp": 0, "fn": 0},
                    "bbox": {"tp": 0, "fp": 0, "fn": 0},
                }
            for key in ("tp", "fp", "fn"):
                aggregate[class_id][eval_type][key] += counts[key]

    def _inc_class_count(self, class_counts, class_id, key):
        if class_id not in class_counts:
            class_counts[class_id] = {"tp": 0, "fp": 0, "fn": 0}
        class_counts[class_id][key] += 1

    def _class_id_to_name(self, class_id):
        for dataset_id, contiguous_id in self._dataset_id_to_contiguous_id.items():
            if contiguous_id == class_id:
                return self._category_id_to_name.get(dataset_id, f"class_{class_id}")
        if class_id < 0:
            return "UNKNOWN"
        return f"class_{class_id}"

    def _positions_in_pred_mask(self, positions, pred, all_positions=None):
        if not hasattr(pred, "pred_masks"):
            return False

        pred_mask = pred.pred_masks[0].numpy()
        h, w = pred_mask.shape

        for x, y in positions:
            if x < 0 or y < 0 or x >= w or y >= h:
                return False
            if not pred_mask[y, x]:
                return False
        
        # If the prediction mask includes any positions that are not in the GT components,
        # we consider it as not fully containing the GT (for TP) and thus a FP
        if all_positions is not None:
            positions_set = set(positions)
            for x, y in all_positions:
                if (x, y) not in positions_set:
                    if x < 0 or y < 0 or x >= w or y >= h:
                        continue  # out-of-image positions cannot be inside the mask
                    if pred_mask[y, x]:
                        return False

        return True

    def _positions_in_pred_bbox(self, positions, pred, all_positions=None):
        if not hasattr(pred, "pred_boxes"):
            return False

        x1, y1, x2, y2 = pred.pred_boxes.tensor[0].tolist()
        for x, y in positions:
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                return False
            
        # If the prediction bbox includes any positions that are not in the GT components,
        # we consider it as not fully containing the GT (for TP) and thus a FP
        if all_positions is not None:
            positions_set = set(positions)
            for x, y in all_positions:
                if (x, y) not in positions_set and (x1 <= x <= x2 and y1 <= y <= y2):
                    return False
        return True
    