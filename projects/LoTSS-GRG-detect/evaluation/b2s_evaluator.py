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
            tp_association,
            fp_association,
            fn_association,
            tp_multicomponent_association,
            fp_multicomponent_association,
            fn_multicomponent_association,
            diagnostics,
        ) = self._evaluate_on_predictions()


        # ---- Membership metrics ----
        membership_jaccard = self._jaccard(tp_membership, fp_membership, fn_membership)
        membership_precision = self._precision(tp_membership, fp_membership)
        membership_recall = self._recall(tp_membership, fn_membership)
        membership_f1 = self._f1(membership_precision, membership_recall)

        # ---- Validity metrics ----
        validity_jaccard = self._jaccard(tp_validity, fp_validity, fn_validity)
        validity_precision = self._precision(tp_validity, fp_validity)
        validity_recall = self._recall(tp_validity, fn_validity)
        validity_f1 = self._f1(validity_precision, validity_recall)

        # ---- Association metrics ----
        association_jaccard = self._jaccard(tp_association, fp_association, fn_association)
        association_precision = self._precision(tp_association, fp_association)
        association_recall = self._recall(tp_association, fn_association)
        association_f1 = self._f1(association_precision, association_recall)

        # ---- Multi-component association metrics ----
        mcs_association_jaccard = self._jaccard(
            tp_multicomponent_association, 
            fp_multicomponent_association, 
            fn_multicomponent_association
        )
        mcs_association_precision = self._precision(
            tp_multicomponent_association, 
            fp_multicomponent_association
        )
        mcs_association_recall = self._recall(
            tp_multicomponent_association, 
            fn_multicomponent_association
        )
        mcs_association_f1 = self._f1(mcs_association_precision, mcs_association_recall)


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
        ]
        association_results_str = [
            "Association Jaccard",
            "Association Precision",
            "Association Recall",
            "Association F1"
        ]
        mcs_association_results_str = [
            "MCS Association Jaccard",
            "MCS Association Precision",
            "MCS Association Recall",
            "MCS Association F1"
        ]
        monitoring_metrics = [
            "Validity GT Positive Rate",
            "Validity Pred Positive Rate",
            "Validity Logit Mean",
            "Validity Logit Std",
            "Validity PR-AUC"
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
            "Association Jaccard": association_jaccard,
            "Association Precision": association_precision,
            "Association Recall": association_recall,
            "Association F1": association_f1,
            "MCS Association Jaccard": mcs_association_jaccard,
            "MCS Association Precision": mcs_association_precision,
            "MCS Association Recall": mcs_association_recall,
            "MCS Association F1": mcs_association_f1,
            "Validity GT Positive Rate": diagnostics["gt_validity_pos_rate"],
            "Validity Pred Positive Rate": diagnostics["pred_validity_pos_rate"],
            "Validity Logit Mean": diagnostics["validity_logit_mean"],
            "Validity Logit Std": diagnostics["validity_logit_std"],
            "Validity PR-AUC": diagnostics["validity_pr_auc"],
        }

        membership_results = {k: results[k] for k in membership_results_str}
        validity_results = {k: results[k] for k in validity_results_str}
        association_results = {k: results[k] for k in association_results_str}
        mcs_association_results = {k: results[k] for k in mcs_association_results_str}
        monitoring_metrics_results = {k: results[k] for k in monitoring_metrics}
        
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
            +
            "\n"
            +
            create_small_table(monitoring_metrics_results)
        )
        self._logger.info(
            "B2S Evaluation Results for Component Association (range 0-1):"
            +
            "\n"
            +
            create_small_table(association_results)
        )
        self._logger.info(
            "B2S Evaluation Results for Multi-component Association (range 0-1):"
            +
            "\n"
            +
            create_small_table(mcs_association_results)
        )
        return copy.deepcopy({
            "B2S": results
        })

    def _evaluate_on_predictions(self):
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

        tp_association_list = []
        fp_association_list = []
        fn_association_list = []

        tp_multicomponent_association_list = []
        fp_multicomponent_association_list = []
        fn_multicomponent_association_list = []
        
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

            gt_valid = np.asarray(gt_proposal_validity).reshape(-1).astype(np.int32)
            gt_member = np.asarray(gt_component_membership).astype(np.int32)
            pred_valid = np.asarray(pred_proposal_validity_binary).reshape(-1).astype(np.int32)
            pred_member = np.asarray(pred_component_membership_binary).astype(np.int32)
    
            tp_association, fp_association, fn_association = self._component_association_evaluation(
                gt_valid=gt_valid,
                gt_member=gt_member,
                pred_valid=pred_valid,
                pred_member=pred_member
            )

            (
                tp_multicomponent_association,
                fp_multicomponent_association,
                fn_multicomponent_association
            ) = self._multicomponent_association_evaluation(
                gt_valid=gt_valid,
                gt_member=gt_member,
                pred_valid=pred_valid,
                pred_member=pred_member
            )

            tp_association_list.append(tp_association)
            fp_association_list.append(fp_association)
            fn_association_list.append(fn_association)

            tp_multicomponent_association_list.append(tp_multicomponent_association)
            fp_multicomponent_association_list.append(fp_multicomponent_association)
            fn_multicomponent_association_list.append(fn_multicomponent_association)

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

        tp_association_list = self._flatten_and_concat(tp_association_list)
        fp_association_list = self._flatten_and_concat(fp_association_list)
        fn_association_list = self._flatten_and_concat(fn_association_list)

        tp_multicomponent_association_list = self._flatten_and_concat(tp_multicomponent_association_list)
        fp_multicomponent_association_list = self._flatten_and_concat(fp_multicomponent_association_list)
        fn_multicomponent_association_list = self._flatten_and_concat(fn_multicomponent_association_list)

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
            tp_association_list,
            fp_association_list,
            fn_association_list,
            tp_multicomponent_association_list,
            fp_multicomponent_association_list,
            fn_multicomponent_association_list,
            diagnostics,
        )
    
    def _component_association_evaluation(self, gt_valid, gt_member, pred_valid, pred_member):
        """
        Evaluate component association performance on the subset
        of proposals which are associated with a real source in GT or prediction.
        A proposal is a true positive only if it is predicted valid and its full
        component membership vector matches the GT vector exactly.

        Wrong membership on a GT-valid proposal counts as both:
        - FP: we predicted an incorrect association
        - FN: we missed the correct association

        This keeps proposal-level association recall from being artificially high
        when validity is correct but membership is wrong.
        """
        if gt_member.ndim == 1:
            gt_member = gt_member.reshape(-1, 1)
        if pred_member.ndim == 1:
            pred_member = pred_member.reshape(-1, 1)

        if gt_valid.shape[0] != pred_valid.shape[0]:
            raise ValueError(
                f"Validity length mismatch: gt={gt_valid.shape[0]}, pred={pred_valid.shape[0]}"
            )
        if gt_member.shape != pred_member.shape:
            raise ValueError(
                "Membership shape mismatch for per-proposal evaluation. "
                f"gt_member={gt_member.shape}, pred_member={pred_member.shape}"
            )
        if gt_member.shape[0] != gt_valid.shape[0]:
            raise ValueError(
                "Membership and validity lengths must match for per-proposal evaluation. "
                f"membership={gt_member.shape[0]}, validity={gt_valid.shape[0]}"
            )

        gt_positive = gt_valid == 1
        pred_positive = pred_valid == 1
        membership_correct = np.all(pred_member == gt_member, axis=1)

        tp = np.logical_and.reduce((
            gt_positive,
            pred_positive,
            membership_correct,
        )).astype(np.int32)

        fp = np.logical_and(
            pred_positive,
            np.logical_or(
                np.logical_not(gt_positive),
                np.logical_not(membership_correct),
            ),
        ).astype(np.int32)

        fn = np.logical_and(
            gt_positive,
            np.logical_or(
                np.logical_not(pred_positive),
                np.logical_not(membership_correct),
            ),
        ).astype(np.int32)

        return tp, fp, fn

    def _multicomponent_association_evaluation(self, gt_valid, gt_member, pred_valid, pred_member):
        """
        Evaluate component association performance on proposals with multiple components
        """
        # We need to look at each proposal and check if the gt members are more
        # than one, and if so, check if the predicted members match the gt members.
        gt_valid = np.asarray(gt_valid).reshape(-1).astype(np.int32)
        pred_valid = np.asarray(pred_valid).reshape(-1).astype(np.int32)
        gt_member = np.asarray(gt_member).astype(np.int32)
        pred_member = np.asarray(pred_member).astype(np.int32)

        gt_multicomponent = np.logical_and(
            gt_valid == 1, 
            np.sum(gt_member, axis=1) > 1
        )
        pred_multicomponent = np.logical_and(
            pred_valid == 1, 
            np.sum(pred_member, axis=1) > 1
        )

        # Include proposals flagged as MCS by either side
        mcs_mask = np.logical_or(gt_multicomponent, pred_multicomponent)
        
        # Masked out all predictions and GT that are not multi-component
        gt_multicomponent_member = gt_member[mcs_mask]
        pred_multicomponent_member = pred_member[mcs_mask]
        gt_multicomponent_valid = gt_valid[mcs_mask]
        pred_multicomponent_valid = pred_valid[mcs_mask]

        # Now we can convert them back to the same format as the single-component
        # evaluation by treating each unique combination of components as a separate class.
        return self._component_association_evaluation(
            gt_valid=gt_multicomponent_valid,
            gt_member=gt_multicomponent_member,
            pred_valid=pred_multicomponent_valid,
            pred_member=pred_multicomponent_member
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


class B2SMultiClassEvaluator(B2SEvaluator):
    """
    Extends B2SEvaluator for three-class proposal validity:
        class 0 = invalid, class 1 = SCS, class 2 = MCS

    Expects pred_proposal_validity to be (N, 3) softmax probabilities
    (produced by _attach_predictions when two_classes=True).
    Expects gt_proposal_validity to be (N,) integer class indices {0, 1, 2}.

    Metrics:
      - Per-class Jaccard, Precision, Recall, F1  (one-vs-rest)
      - Micro accuracy and Macro F1
      - Cohen's Kappa            — near 0: model just learned the class prior
      - GT vs predicted class rates — direct overfitting signal (pred rate >> GT rate)
      - Normalized prediction entropy — near 0: model always collapses to one class
    """

    CLASS_NAMES = ["invalid", "SCS", "MCS"]

    def __init__(self, membership_threshold: float = 0.5):
        # validity_threshold is not used for argmax classification
        super().__init__(validity_threshold=0.5, membership_threshold=membership_threshold)
        self._logger = setup_logger(
            name="LoTSS-B2S-detect.evaluation.B2SMultiClassEvaluator",
            termcolor="cyan",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self):
        if len(self._predictions) == 0:
            self._logger.warning("[B2SMultiClassEvaluator] Did not receive valid predictions.")
            return {}

        (
            tp_membership,
            fp_membership,
            fn_membership,
            tp_validity,
            fp_validity,
            fn_validity,
            per_class_tp,
            per_class_fp,
            per_class_fn,
            diagnostics,
        ) = self._evaluate_on_predictions()

        # ---- Membership (identical to parent) -------------------------
        membership_jaccard   = self._jaccard(tp_membership, fp_membership, fn_membership)
        membership_precision = self._precision(tp_membership, fp_membership)
        membership_recall    = self._recall(tp_membership, fn_membership)
        membership_f1        = self._f1(membership_precision, membership_recall)

        # ---- Validity (identical to parent) -------------------------
        validity_jaccard   = self._jaccard(tp_validity, fp_validity, fn_validity)
        validity_precision = self._precision(tp_validity, fp_validity)
        validity_recall    = self._recall(tp_validity, fn_validity)
        validity_f1        = self._f1(validity_precision, validity_recall)

        # ---- Per-class validity metrics (one-vs-rest) -----------------
        per_class_metrics = {}
        per_class_f1s = []
        for k, name in enumerate(self.CLASS_NAMES):
            p  = self._precision(per_class_tp[k], per_class_fp[k])
            r  = self._recall(per_class_tp[k], per_class_fn[k])
            f1 = self._f1(p, r)
            j  = self._jaccard(per_class_tp[k], per_class_fp[k], per_class_fn[k])
            per_class_metrics[name] = {"Jaccard": j, "Precision": p, "Recall": r, "F1": f1}
            per_class_f1s.append(f1)

        macro_f1 = float(np.mean(per_class_f1s))

        # ---- Flat results dict (detectron2-compatible) ----------------
        results = {
            # Membership metrics
            "Membership Jaccard":   membership_jaccard,
            "Membership Precision": membership_precision,
            "Membership Recall":    membership_recall,
            "Membership F1":        membership_f1,

            # Validity metrics
            "Validity Jaccard":     validity_jaccard,
            "Validity Precision":   validity_precision,
            "Validity Recall":      validity_recall,
            "Validity F1":          validity_f1,

            # Global validity diagnostics
            # (not really metrics, but important for
            # interpreting results and diagnosing overfitting)
            "Validity Micro Accuracy": diagnostics["micro_accuracy"],
            "Validity Macro F1":       macro_f1,
            "Validity Cohen Kappa":    diagnostics["cohen_kappa"],
            "Validity Norm Entropy":   diagnostics["norm_pred_entropy"],
        }
        for name in self.CLASS_NAMES:
            for metric, val in per_class_metrics[name].items():
                results[f"Validity {name} {metric}"] = val
        for name in self.CLASS_NAMES:
            results[f"GT Rate {name}"]   = diagnostics["gt_class_rates"][name]
            results[f"Pred Rate {name}"] = diagnostics["pred_class_rates"][name]

        # ---- Logging --------------------------------------------------
        self._logger.info(
            "B2S Multi-Class Evaluation - Component Membership:\n"
            + create_small_table({
                "Jaccard":   membership_jaccard,
                "Precision": membership_precision,
                "Recall":    membership_recall,
                "F1":        membership_f1,
            })
        )
        self._logger.info(
            "B2S Multi-Class Evaluation - Proposal Validity:\n"
            + create_small_table({
                "Jaccard":   validity_jaccard,
                "Precision": validity_precision,
                "Recall":    validity_recall,
                "F1":        validity_f1,
            })
        )
        for name in self.CLASS_NAMES:
            self._logger.info(
                f"B2S Multi-Class Evaluation - Validity [{name}]:\n"
                + create_small_table(per_class_metrics[name])
            )

        self._logger.info(
            "B2S Multi-Class Evaluation - Validity Aggregate:\n"
            + create_small_table({
                "Micro Accuracy": diagnostics["micro_accuracy"],
                "Macro F1":       macro_f1,
                "Cohen Kappa":    diagnostics["cohen_kappa"],
            })
        )
        # Class balance table: most important overfitting diagnostic.
        # If Pred Rate for a class >> GT Rate, the model has collapsed onto it.
        # Normalized entropy near 0 confirms the collapse.
        balance_table = {}
        for name in self.CLASS_NAMES:
            balance_table[f"GT {name}"]   = diagnostics["gt_class_rates"][name]
            balance_table[f"Pred {name}"] = diagnostics["pred_class_rates"][name]
        balance_table["Norm Entropy (0=collapse, 1=uniform)"] = diagnostics["norm_pred_entropy"]
        self._logger.info(
            "B2S Multi-Class Evaluation - Class Balance (overfitting signal):\n"
            + create_small_table(balance_table)
        )

        return copy.deepcopy({"B2S": results})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_on_predictions(self):
        tp_membership_list = []
        fp_membership_list = []
        fn_membership_list = []

        tp_validity_list = []
        fp_validity_list = []
        fn_validity_list = []

        per_class_tp_lists = [[] for _ in range(3)]
        per_class_fp_lists = [[] for _ in range(3)]
        per_class_fn_lists = [[] for _ in range(3)]

        gt_class_list   = []
        pred_class_list = []
        pred_probs_list = []  # (N, 3) softmax probs — used for entropy

        for prediction in self._predictions:
            gt_proposal_validity   = self._to_numpy(prediction["gt_proposal_validity"]) # (N,)
            gt_component_membership = self._to_numpy(prediction["gt_component_membership"]) # (N, C)

            instances = prediction["instances"]
            pred_proposal_validity  = self._to_numpy(instances.pred_proposal_validity)   # (N, 3)
            pred_component_membership = self._to_numpy(instances.pred_component_membership)  # (N, C)

            # No need for thresholding here since we are doing argmax classification for validity
            # We are going to set all predicted classes that are not the invalid class (0)
            # to 1, and also do this with the ground truth for consistency in TP/FP/FN calculations.
            # This allows us to gauge how well the model distinguishes valid (SCS/MCS) vs invalid proposals,
            # which is the most important aspect of validity prediction.
            pred_class = np.argmax(pred_proposal_validity, axis=-1).astype(np.int32) # (N,)
            pred_class_binary = np.where(pred_class == 0, 0, 1)  # Convert to binary for TP/FP/FN (0=invalid, 1=valid)
            gt_proposal_validity_binary = np.where(gt_proposal_validity == 0, 0, 1).astype(np.int32)  # Convert to binary for TP/FP
            
            # We treat membership the same as in the binary case,
            # since it's still a multi-label prediction (multiple components can belong to a proposal).
            pred_component_membership_binary = (pred_component_membership >= self._membership_threshold).astype(np.int32)  # (N, C)

            tp_membership_list.append(self._tp(pred_component_membership_binary, gt_component_membership))
            fp_membership_list.append(self._fp(pred_component_membership_binary, gt_component_membership))
            fn_membership_list.append(self._fn(pred_component_membership_binary, gt_component_membership))
            tp_validity_list.append(self._tp(pred_class_binary, gt_proposal_validity_binary))
            fp_validity_list.append(self._fp(pred_class_binary, gt_proposal_validity_binary))
            fn_validity_list.append(self._fn(pred_class_binary, gt_proposal_validity_binary))

            gt_class_list.append(gt_proposal_validity)
            pred_class_list.append(pred_class)
            pred_probs_list.append(pred_proposal_validity.reshape(-1, 3))

            for k in range(3):
                gt_k   = (gt_proposal_validity == k).astype(np.int32)
                pred_k = (pred_class == k).astype(np.int32)
                per_class_tp_lists[k].append(self._tp(pred_k, gt_k))
                per_class_fp_lists[k].append(self._fp(pred_k, gt_k))
                per_class_fn_lists[k].append(self._fn(pred_k, gt_k))

        tp_membership = self._flatten_and_concat(tp_membership_list)
        fp_membership = self._flatten_and_concat(fp_membership_list)
        fn_membership = self._flatten_and_concat(fn_membership_list)

        tp_validity = self._flatten_and_concat(tp_validity_list)
        fp_validity = self._flatten_and_concat(fp_validity_list)
        fn_validity = self._flatten_and_concat(fn_validity_list)

        per_class_tp = [self._flatten_and_concat(per_class_tp_lists[k]) for k in range(3)]
        per_class_fp = [self._flatten_and_concat(per_class_fp_lists[k]) for k in range(3)]
        per_class_fn = [self._flatten_and_concat(per_class_fn_lists[k]) for k in range(3)]

        gt_class   = np.concatenate(gt_class_list)   if gt_class_list   else np.array([], dtype=np.int32)
        pred_class = np.concatenate(pred_class_list) if pred_class_list else np.array([], dtype=np.int32)
        pred_probs = np.concatenate(pred_probs_list, axis=0) if pred_probs_list else np.zeros((0, 3), dtype=np.float32)

        n = len(gt_class)
        gt_class_rates   = {name: float((gt_class   == k).mean()) if n > 0 else 0.0 for k, name in enumerate(self.CLASS_NAMES)}
        pred_class_rates = {name: float((pred_class == k).mean()) if n > 0 else 0.0 for k, name in enumerate(self.CLASS_NAMES)}

        micro_accuracy = float((gt_class == pred_class).mean()) if n > 0 else 0.0
        kappa          = self._cohen_kappa(gt_class, pred_class, n_classes=3)

        # Prediction entropy per sample, normalised to [0, 1] where:
        #   0 = model always picks the same class  (collapsed / overfit)
        #   1 = model is perfectly uncertain about all classes
        eps = 1e-9
        entropy     = -np.sum(pred_probs * np.log(pred_probs + eps), axis=-1)  # (N,)
        max_entropy = np.log(3)
        norm_entropy = float(entropy.mean() / max_entropy) if n > 0 else 0.0

        diagnostics = {
            "gt_class_rates":   gt_class_rates,
            "pred_class_rates": pred_class_rates,
            "micro_accuracy":   micro_accuracy,
            "cohen_kappa":      kappa,
            "norm_pred_entropy": norm_entropy,
        }

        return (
            tp_membership,
            fp_membership,
            fn_membership,
            tp_validity,
            fp_validity,
            fn_validity,
            per_class_tp,
            per_class_fp,
            per_class_fn,
            diagnostics,
        )

    @staticmethod
    def _cohen_kappa(gt: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
        """
        Cohen's Kappa corrected for chance agreement.
        κ ≈ 1  → near-perfect agreement
        κ ≈ 0  → model performs at chance level (just learned the class prior)
        κ < 0  → model is systematically wrong
        """
        n = len(gt)
        if n == 0:
            return 0.0
        p_o = float(np.sum(gt == pred)) / n
        p_e = sum(
            (float(np.sum(gt == k)) / n) * (float(np.sum(pred == k)) / n)
            for k in range(n_classes)
        )
        denom = 1.0 - p_e
        return (p_o - p_e) / denom if denom > 1e-9 else 0.0
