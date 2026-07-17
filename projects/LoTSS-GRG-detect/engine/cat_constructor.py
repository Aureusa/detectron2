import random
import numpy as np

from scipy.optimize import linear_sum_assignment


class PredictionObject:
    def __init__(self, prediction):
        self.prediction = prediction

    def get_pred_classes(self):
        pred_classes = self.prediction.pred_classes
        pred_class = "MCS" if int(pred_classes.item()) == 1 else "SCS"
        return pred_class

    def get_pred_masks(self):
        pred_masks = self.prediction.pred_masks[0].cpu().numpy()
        return pred_masks

    def get_pred_boxes(self):
        prediction = self.prediction.pred_boxes
        x1, y1, x2, y2 = prediction.tensor[0].tolist()
        return x1, y1, x2, y2
    
    def get_pred_scores(self):
        pred_scores = self.prediction.scores[0].item()
        return pred_scores


class CatalogueConstructor:   
    def build_gt_catalogue(self, image_id: int, image_metadata: dict, anns: list[dict]):
        """
        Treats each GT annotations as a source, and assigns components to it based on the instance_positions.
        Returns a dict: source_name -> {"components": set(...), "class": ..., "score": None, "image_id": ...}

        :param image_id: ID of the image
        :type image_id: int
        :param image_metadata: Metadata of the image, containing component and source info
        :type image_metadata: dict
        :param anns: List of GT annotations for the image, each with "category_id" and "instance_positions"
        :type anns: list[dict]
        """
        pos_map = self._build_position_to_component_map(image_metadata)
        sources = {}

        for ann in anns:
            components = set()
            class_name = "MCS" if ann["category_id"] == 2 else "SCS"
            source_name = None
            for pos in ann["instance_positions"]:
                comp_info = pos_map.get(tuple(pos))
                if comp_info is None:
                    continue
                if source_name is None:
                    source_name = comp_info["source_name"]
                
                component_name = comp_info["component_name"]
                components.add(component_name)

            if source_name is not None:
                sources[source_name] = {
                    "components": components,
                    "class": class_name,
                    "score": None, # for GT, score is not applicable but kept for consistency with pred format
                    "image_id": image_id,
                }
        return sources

    def build_pred_catalogue(self, image_id: int, image_metadata: dict, pred_instances: list, mask: bool = False):
        """
        For each predicted instance, determine which components it covers (either via mask or bounding box),
        which gives a source with assigned components, predicted class, and score.

        :param image_id: ID of the image
        :type image_id: int
        :param image_metadata: Metadata of the image, containing component and source info
        :type image_metadata: dict
        :param pred_instances: List of predicted instances for the image
        :type pred_instances: list
        :param mask: Whether to use masks or bounding boxes for component assignment
        :type mask: bool
        """
        pos_map = self._build_position_to_component_map(image_metadata)
        sources = {}

        if pred_instances is None:
            return sources
        
        for pred_idx in range(len(pred_instances)): # Loop over each predicted instance == source
            pred = pred_instances[pred_idx]
            pred_obj = PredictionObject(pred)
            if mask:
                components, pred_class, pred_score = self._build_pred_catalogue_with_masks(pred_obj, pos_map)
            else:
                components, pred_class, pred_score = self._build_pred_catalogue_with_boxes(pred_obj, pos_map)
            source_name = f"pred_source_{pred_idx}"
            sources[source_name] = {
                "components": components,
                "class": pred_class,
                "score": pred_score,
                "image_id": image_id,
            }
        return sources

    def compare_catalogues(self, gt_cat: dict, pred_cat: dict) -> dict:
        """
        Compare GT and predicted catalogues at the source level.
        """
        set_gt = self._set_constructor(gt_cat)
        set_pred = self._set_constructor(pred_cat)

        C, c_array = self._similarity_matrix(set_gt, set_pred)
        opt_assignment = self._hungarian_algorithm(set_gt, set_pred, C, c_array)

        results, aggregated_counts_scs, aggregated_counts_mcs = self._compare_catalogues_sourcewise(
            opt_assignment, set_gt, set_pred
        )
        return results, aggregated_counts_scs, aggregated_counts_mcs

    def _compare_catalogues_sourcewise(self, opt_assignment: dict, set_gt: dict, set_pred: dict) -> dict:
        """Compare GT and predicted catalogues at the source level"""
        aggregated_counts_scs = {
            "tp_source": 0,
            "fp_source": 0,
            "fn_source": 0,
            "tp_component": 0,
            "fp_component": 0,
            "fn_component": 0,
            "Perfect match": 0,
            "Near-perfect match": 0,
            "Partial match": 0,
            "Wrong class": 0,
            "No match": 0,
            "Missed source": 0,
            "Hallucinated source": 0,
        }
        aggregated_counts_mcs = {
            "tp_source": 0,
            "fp_source": 0,
            "fn_source": 0,
            "tp_component": 0,
            "fp_component": 0,
            "fn_component": 0,
            "Perfect match": 0,
            "Near-perfect match": 0,
            "Partial match": 0,
            "Wrong class": 0,
            "No match": 0,
            "Missed source": 0,
            "Hallucinated source": 0,
        }
        results = {}
        for source_name, assign_info in opt_assignment.items():
            similarity = assign_info["similarity"]
            gt_class = assign_info["gt_class"]
            pred_class = assign_info["pred_class"]
            num_gt_components = assign_info["num_gt_components"]
            num_pred_components = assign_info["num_pred_components"]
            num_correct_components = assign_info["num_correct_components"]
            num_missed_components = assign_info["num_missed_components"]
            num_hallucinated_components = assign_info["num_hallucinated_components"]
            pred_score = assign_info["pred_score"]

            if similarity == 1.0 and gt_class == pred_class:
                source_status = "TP"
                reason = "Perfect match"
                self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "tp_source", pred_class)
                self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "Perfect match", pred_class)
            elif similarity >= 0.8 and gt_class == pred_class:
                source_status = "FP"
                reason = "Near-perfect match"
                self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "fp_source", pred_class)
                self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "Near-perfect match", pred_class)
            elif similarity > 0.0 and gt_class == pred_class:
                source_status = "FP"
                reason = "Partial match"
                self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "fp_source", pred_class)
                self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "Partial match", pred_class)
            elif similarity > 0.0 and gt_class != pred_class:
                source_status = "FP"
                reason = "Wrong class"
                self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "fp_source", pred_class)
                self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "Wrong class", pred_class)
            elif similarity == 0.0:
                source_status = "FN"
                reason = "No match"
                self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "fn_source", gt_class)
                self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "No match", gt_class)

            componentwise = {
                "tp": num_correct_components, # what I predicted that is real and correct
                "fn": num_missed_components, # what I missed that is real
                "fp": num_hallucinated_components # what I predicted that isn’t real
            }
            self._aggregate_counts(
                aggregated_counts_scs,
                aggregated_counts_mcs,
                num_correct_components,
                "tp_component",
                pred_class
            )
            self._aggregate_counts(
                aggregated_counts_scs,
                aggregated_counts_mcs,
                num_missed_components,
                "fn_component",
                pred_class
            )
            self._aggregate_counts(
                aggregated_counts_scs,
                aggregated_counts_mcs,
                num_hallucinated_components,
                "fp_component",
                pred_class
            )

            results[source_name] = {
                "similarity": similarity,
                "gt_class": gt_class,
                "pred_class": pred_class,
                "source_status": source_status,
                "reason": reason,
                "num_gt_components": num_gt_components,
                "num_pred_components": num_pred_components,
                "num_correct_components": num_correct_components,
                "num_missed_components": num_missed_components,
                "num_hallucinated_components": num_hallucinated_components,
                "componentwise": componentwise,
                "pred_score": pred_score,
            }

        # If not all sources were assigned,
        # we can consider unassigned GT sources as FN and unassigned pred sources as FP
        assigned_gt_sources = set(results.keys())
        assigned_pred_sources = {info["assigned_to"] for info in opt_assignment.values()}
        all_gt_sources = set(set_gt["set"].keys())
        all_pred_sources = set(set_pred["set"].keys())

        unassigned_gt_sources = all_gt_sources - assigned_gt_sources
        unassigned_pred_sources = all_pred_sources - assigned_pred_sources

        for gt_source in unassigned_gt_sources:
            pred_class = set_gt["class"][gt_source]
            results[gt_source] = {
                "similarity": 0.0,
                "gt_class": set_gt["class"][gt_source],
                "pred_class": None,
                "source_status": "FN",
                "reason": "Missed source",
                "num_gt_components": len(set_gt["set"][gt_source]),
                "num_pred_components": 0,
                "num_correct_components": 0,
                "num_missed_components": len(set_gt["set"][gt_source]),
                "pred_score": 0.0,
            }
            self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "fn_source", pred_class)
            self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "Missed source", pred_class)

            # Contribute FN to componentwise counts
            self._aggregate_counts(
                aggregated_counts_scs,
                aggregated_counts_mcs,
                len(set_gt["set"][gt_source]),
                "fn_component",
                pred_class
            )

        for pred_source in unassigned_pred_sources:
            pred_class = set_pred["class"][pred_source]
            results[pred_source] = {
                "similarity": 0.0,
                "gt_class": None,
                "pred_class": set_pred["class"][pred_source],
                "source_status": "FP",
                "reason": "Hallucinated source",
                "num_gt_components": 0,
                "num_pred_components": len(set_pred["set"][pred_source]),
                "num_correct_components": 0,
                "num_missed_components": 0,
                "pred_score": set_pred["score"][pred_source],
            }
            self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "fp_source", pred_class)
            self._aggregate_counts(aggregated_counts_scs, aggregated_counts_mcs, 1, "Hallucinated source", pred_class)

            # Contribute FP to componentwise counts
            self._aggregate_counts(
                aggregated_counts_scs,
                aggregated_counts_mcs,
                len(set_pred["set"][pred_source]),
                "fp_component",
                pred_class
            )
        return results, aggregated_counts_scs, aggregated_counts_mcs
    
    def _aggregate_counts(
            self,
            container_scs: dict,
            container_mcs: dict,
            num: int, field: str,
            pred_class: str
        ) -> None:
        """
        Helper to aggregate TP/FP/FN counts. It adds num to container[field], initializing it to 0 if not present.
        Performs in-place update of container.
        """
        if pred_class == "SCS":
            container = container_scs
        else:
            container = container_mcs

        if field not in container:
            container[field] = 0
        container[field] += num

    def _set_constructor(self, cat: dict) -> dict:
        set_ = {"set": {}, "class": {}, "score": {}}
        for source_name, info in cat.items():
            components = info["components"]
            class_ = info["class"]
            score = info["score"]
            set_["set"][source_name] = components
            set_["class"][source_name] = class_
            set_["score"][source_name] = score
        return set_
    
    def _similarity_matrix(self, set_gt: dict, set_pred: dict) -> dict:
        P = set_pred["set"]
        G = set_gt["set"]

        # Create a similarity matrix C of size N x M
        C = {
            "similarity_matrix": {},
            "pred_comp": {},
            "gt_comp": {},
            "correct_comp": {},
            "missed_comp": {},
            "hallucinated_comp": {}
        }
        
        for g_key, g_set in G.items():
            C["similarity_matrix"][g_key] = {}
            C["pred_comp"][g_key] = {}
            C["gt_comp"][g_key] = {}
            C["correct_comp"][g_key] = {}
            C["missed_comp"][g_key] = {}
            C["hallucinated_comp"][g_key] = {}
            for p_key, p_set in P.items():
                # Calculate the similarity between g_set and p_set
                intersection = g_set.intersection(p_set)
                union = g_set.union(p_set)
                
                num_intersection = len(intersection)
                num_union = len(union)
                if num_union > 0:
                    similarity = num_intersection / num_union
                else:
                    similarity = 0.0

                missed_comp = len(g_set - p_set)
                hallucinated_comp = len(p_set - g_set)
                gt_comp = len(g_set)
                pred_comp = len(p_set)

                # How many components in GT are covered by the prediction? (recall-like)
                C["similarity_matrix"][g_key][p_key] = similarity
                C["pred_comp"][g_key][p_key] = pred_comp
                C["gt_comp"][g_key][p_key] = gt_comp
                C["correct_comp"][g_key][p_key] = num_intersection
                C["missed_comp"][g_key][p_key] = missed_comp
                C["hallucinated_comp"][g_key][p_key] = hallucinated_comp
        
        c_array = np.array([[C["similarity_matrix"][g_key][p_key] for p_key in P.keys()] for g_key in G.keys()])
        return C, c_array
    
    def _hungarian_algorithm(self, set_gt: dict, set_pred: dict, C, c_array: np.ndarray) -> dict:
        row_ind, col_ind = linear_sum_assignment(-c_array)
        
        g_keys = list(set_gt["set"].keys())
        p_keys = list(set_pred["set"].keys())
        opt_assignment = {}
        for g_idx, p_idx in zip(row_ind, col_ind):
            g_key = g_keys[g_idx]
            p_key = p_keys[p_idx]

            opt_assignment[g_key] = {
                "assigned_to": p_key,
                "similarity": C["similarity_matrix"][g_key][p_key],
                "num_gt_components": C["gt_comp"][g_key][p_key],
                "num_pred_components": C["pred_comp"][g_key][p_key],
                "num_correct_components": C["correct_comp"][g_key][p_key],
                "num_missed_components": C["missed_comp"][g_key][p_key],
                "num_hallucinated_components": C["hallucinated_comp"][g_key][p_key],
                "gt_class": set_gt["class"][g_key],
                "pred_class": set_pred["class"][p_key],
                "pred_score": set_pred["score"][p_key],
            }
        return opt_assignment
    
    def _build_pred_catalogue_with_masks(self, predictions: PredictionObject, pos_map: dict):
        mask = predictions.get_pred_masks()
        pred_class = predictions.get_pred_classes()
        pred_score = predictions.get_pred_scores()

        # Check which positions fall inside the mask
        components = set()
        for pos, comp_info in pos_map.items():
            x, y = pos
            if mask[y, x]:  # Assuming mask is a 2D array where True indicates the predicted area
                components.add(comp_info["component_name"])
        return components, pred_class, pred_score
    
    def _build_pred_catalogue_with_boxes(self, predictions: PredictionObject, pos_map: dict):
        x1, y1, x2, y2 = predictions.get_pred_boxes()
        pred_class = predictions.get_pred_classes()
        pred_score = predictions.get_pred_scores()

        # Check which positions fall inside the bounding box
        components = set()
        for pos, comp_info in pos_map.items():
            x, y = pos
            if x1 <= x <= x2 and y1 <= y <= y2:
                components.add(comp_info["component_name"])
        return components, pred_class, pred_score
    
    def _resolve_duplicate_predictions(self, pred_rows: list[dict]) -> list[dict]:
        """
        If multiple predictions claim the same component, keep only the one with the highest score.
        """
        best_pred_for_comp = {}
        for row in pred_rows:
            comp = row["comp_name"]
            score = row["pred_score"]
            if comp not in best_pred_for_comp or score > best_pred_for_comp[comp]["pred_score"]:
                best_pred_for_comp[comp] = row
        return list(best_pred_for_comp.values())
    
    def _build_position_to_component_map(self, image_metadata):
        """
        Returns a dict: (x, y) -> {"component_name": ..., "source_name": ..., "source_id": ...}
        """
        components = image_metadata["candidates"]["grouping"]["components"]
        return {
            tuple(c["xy"]): {
                "component_name": c["component_name"],
                "source_name": c["source_name"],
                "source_id": c["source_id"],
            }
            for c in components
        }
    