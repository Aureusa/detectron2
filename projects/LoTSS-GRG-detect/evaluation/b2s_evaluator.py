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


class B2SEvaluator(DatasetEvaluator):
    def __init__(self, validity_threshold: float = 0.5, membership_threshold: float = 0.5):
        self._cpu_device = torch.device("cpu")
        self._validity_threshold = validity_threshold
        self._membership_threshold = membership_threshold
        self._logger = setup_logger(name="LoTSS-B2S-detect.evaluation.B2SEvaluator", termcolor="magenta")

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
            anns = input_data.get("annotations", None)[0] if "annotations" in input_data else None
            if anns is None:
                raise ValueError("Input data must contain 'annotations' key for evaluation.")
            
            prediction = {
                "image_id": input_data["image_id"],
                "instances": output["instances"].to(self._cpu_device) if "instances" in output else None,
                "gt_proposal_validity": (
                    anns.get("gt_proposal_validity", None)
                    if anns.get("gt_proposal_validity", None) is not None
                    else None
                ),
                "gt_component_membership": (
                    anns.get("gt_component_membership", None)
                    if anns.get("gt_component_membership", None) is not None
                    else None
                ),
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
            self._logger.warning("[B2SEvaluator] Did not receive valid predictions.")
            return {}
        
        (
            tp_membership,
            fp_membership,
            fn_membership,
            tp_validity,
            fp_validity,
            fn_validity,
            diagnostics,
        ) = self._gather_predictions()


        # Metrics
        membership_jaccard = self._jaccard(tp_membership, fp_membership, fn_membership)
        membership_precision = self._precision(tp_membership, fp_membership)
        membership_recall = self._recall(tp_membership, fn_membership)
        membership_f1 = self._f1(membership_precision, membership_recall)

        validity_jaccard = self._jaccard(tp_validity, fp_validity, fn_validity)
        validity_precision = self._precision(tp_validity, fp_validity)
        validity_recall = self._recall(tp_validity, fn_validity)
        validity_f1 = self._f1(validity_precision, validity_recall)

        membership_results_str = [
            "Membership Jaccard",
            "Membership Precision",
            "Membership Recall",
            "Membership F1"
        ]
        validity_results_str = [
            "Validity Jaccard",
            "Validity Precision",
            "Validity Recall",
            "Validity F1",
            "Validity GT Positive Rate",
            "Validity Pred Positive Rate",
            "Validity Logit Mean",
            "Validity Logit Std",
            "Validity PR-AUC",
        ]

        # Aggregate metrics
        results = {
            "Membership Jaccard": membership_jaccard,
            "Membership Precision": membership_precision,
            "Membership Recall": membership_recall,
            "Membership F1": membership_f1,
            "Validity Jaccard": validity_jaccard,
            "Validity Precision": validity_precision,
            "Validity Recall": validity_recall,
            "Validity F1": validity_f1,
            "Validity GT Positive Rate": diagnostics["gt_validity_pos_rate"],
            "Validity Pred Positive Rate": diagnostics["pred_validity_pos_rate"],
            "Validity Logit Mean": diagnostics["validity_logit_mean"],
            "Validity Logit Std": diagnostics["validity_logit_std"],
            "Validity PR-AUC": diagnostics["validity_pr_auc"],
        }

        membership_results = {k: results[k] for k in membership_results_str}
        validity_results = {k: results[k] for k in validity_results_str}
        
        # Log the results in a nice table format
        self._logger.info(
            "B2S Evaluation Results for Component Membership (range 0-1):"
            +
            "\n"
            +
            create_small_table(membership_results)
        )
        self._logger.info(
            "B2S Evaluation Results for Proposal Validity (range 0-1):"
            +
            "\n"
            +
            create_small_table(validity_results)
        )
        return copy.deepcopy({
            "B2S": results
        })

    def _gather_predictions(self):
        tp_membership_list = []
        fp_membership_list = []
        fn_membership_list = []
        tp_validity_list = []
        fp_validity_list = []
        fn_validity_list = []
        gt_validity_list = []
        pred_validity_binary_list = []
        pred_validity_probs_list = []
        pred_validity_logits_list = []
        
        for prediction in self._predictions:
            gt_proposal_validity = self._to_numpy(prediction["gt_proposal_validity"])  # (N,)
            gt_component_membership = self._to_numpy(prediction["gt_component_membership"])  # (N, C)
            
            instances = prediction["instances"]
            pred_proposal_validity = self._to_numpy(instances.pred_proposal_validity)  # (N,)
            if hasattr(instances, "pred_proposal_validity_logits"):
                pred_proposal_validity_logits = self._to_numpy(instances.pred_proposal_validity_logits)
            else:
                pred_proposal_validity_clamped = np.clip(pred_proposal_validity, 1e-6, 1.0 - 1e-6)
                pred_proposal_validity_logits = np.log(
                    pred_proposal_validity_clamped / (1.0 - pred_proposal_validity_clamped)
                )
            pred_component_membership = self._to_numpy(instances.pred_component_membership)  # (N, C)

            # Set a threshold to convert predicted probabilities to binary predictions
            pred_proposal_validity_binary = (pred_proposal_validity >= self._validity_threshold).astype(np.int32)  # (N,)
            pred_component_membership_binary = (pred_component_membership >= self._membership_threshold).astype(np.int32)  # (N, C)

            gt_validity_list.append(np.asarray(gt_proposal_validity).reshape(-1))
            pred_validity_binary_list.append(np.asarray(pred_proposal_validity_binary).reshape(-1))
            pred_validity_probs_list.append(np.asarray(pred_proposal_validity).reshape(-1))
            pred_validity_logits_list.append(np.asarray(pred_proposal_validity_logits).reshape(-1))

            tp_membership_list.append(self._tp(pred_component_membership_binary, gt_component_membership))
            fp_membership_list.append(self._fp(pred_component_membership_binary, gt_component_membership))
            fn_membership_list.append(self._fn(pred_component_membership_binary, gt_component_membership))
            tp_validity_list.append(self._tp(pred_proposal_validity_binary, gt_proposal_validity))
            fp_validity_list.append(self._fp(pred_proposal_validity_binary, gt_proposal_validity))
            fn_validity_list.append(self._fn(pred_proposal_validity_binary, gt_proposal_validity))

        # Different images can have different proposal/component counts.
        # Flatten per-image arrays and concatenate to get homogeneous 1D arrays.
        tp_membership_list = self._flatten_and_concat(tp_membership_list)
        fp_membership_list = self._flatten_and_concat(fp_membership_list)
        fn_membership_list = self._flatten_and_concat(fn_membership_list)
        tp_validity_list = self._flatten_and_concat(tp_validity_list)
        fp_validity_list = self._flatten_and_concat(fp_validity_list)
        fn_validity_list = self._flatten_and_concat(fn_validity_list)

        gt_validity = self._flatten_and_concat(gt_validity_list)
        pred_validity_binary = self._flatten_and_concat(pred_validity_binary_list)
        pred_validity_probs = self._flatten_and_concat_float(pred_validity_probs_list)
        pred_validity_logits = self._flatten_and_concat_float(pred_validity_logits_list)

        diagnostics = {
            "gt_validity_pos_rate": float(gt_validity.mean()) if gt_validity.size > 0 else 0.0,
            "pred_validity_pos_rate": float(pred_validity_binary.mean()) if pred_validity_binary.size > 0 else 0.0,
            "validity_logit_mean": float(pred_validity_logits.mean()) if pred_validity_logits.size > 0 else 0.0,
            "validity_logit_std": float(pred_validity_logits.std()) if pred_validity_logits.size > 0 else 0.0,
            "validity_pr_auc": self._pr_auc(gt_validity, pred_validity_probs),
        }
        
        return (
            tp_membership_list,
            fp_membership_list,
            fn_membership_list,
            tp_validity_list,
            fp_validity_list,
            fn_validity_list,
            diagnostics,
        )
    
    def _jaccard(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray):
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

    def _to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _flatten_and_concat(self, arrays):
        if len(arrays) == 0:
            return np.array([], dtype=np.int32)
        return np.concatenate([np.asarray(a).reshape(-1) for a in arrays]).astype(np.int32)

    def _flatten_and_concat_float(self, arrays):
        if len(arrays) == 0:
            return np.array([], dtype=np.float32)
        return np.concatenate([np.asarray(a).reshape(-1) for a in arrays]).astype(np.float32)

    def _pr_auc(self, gt: np.ndarray, scores: np.ndarray) -> float:
        if gt.size == 0 or scores.size == 0:
            return 0.0

        gt = np.asarray(gt).reshape(-1).astype(np.int32)
        scores = np.asarray(scores).reshape(-1).astype(np.float32)
        positives = np.sum(gt == 1)
        if positives == 0:
            return 0.0

        order = np.argsort(-scores)
        gt_sorted = gt[order]

        tp = np.cumsum(gt_sorted == 1)
        fp = np.cumsum(gt_sorted == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / positives

        precision = np.concatenate(([1.0], precision))
        recall = np.concatenate(([0.0], recall))
        return float(np.trapz(precision, recall))

    def _tp(self, pred, gt):
        """All predictions which are correct"""
        return np.logical_and(pred == 1, gt == 1).astype(int)

    def _fp(self, pred, gt):
        """All predictions which are incorrect"""
        return np.logical_and(pred == 1, gt == 0).astype(int)

    def _fn(self, pred, gt):
        """All missed predictions"""
        return np.logical_and(pred == 0, gt == 1).astype(int)
