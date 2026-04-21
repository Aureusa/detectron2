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
    def build_gt_catalogue(self, image_id, image_metadata, anns):
        """
        Returns a list of rows:
        {"comp_name": ..., "source_name": ..., "class": ..., "image_id": ...}
        """
        pos_map = self._build_position_to_component_map(image_metadata)
        rows = []

        for ann in anns:
            class_name = "MCS" if ann["category_id"] == 2 else "SCS"
            source_name = None
            for pos in ann["instance_positions"]:
                comp_info = pos_map.get(tuple(pos))
                if comp_info is None:
                    continue
                if source_name is None:
                    source_name = comp_info["source_name"]
                rows.append({
                    "comp_name": comp_info["component_name"],
                    "source_name": comp_info["source_name"],
                    "class": class_name,
                    "pred_idx": -1,  # GT rows have pred_idx of -1
                    "pred_score": -1,  # GT rows have pred_score of -1
                    "image_id": image_id,
                })
        return rows

    def build_pred_catalogue(self, image_id, image_metadata, pred_instances, mask: bool = False):
        """
        For each prediction, find which GT annotation it matches (by containment),
        then emit one row per component position that falls inside the prediction.
        """
        pos_map = self._build_position_to_component_map(image_metadata)
        rows = []

        if pred_instances is None:
            return rows
        
        for pred_idx in range(len(pred_instances)):
            pred = pred_instances[pred_idx]
            pred_obj = PredictionObject(pred)
            if mask:
                curr_rows = self._build_pred_catalogue_with_masks(image_id, pred_idx, pred_obj, pos_map)
            else:
                curr_rows = self._build_pred_catalogue_with_boxes(image_id, pred_idx, pred_obj, pos_map)
            rows.extend(curr_rows)
        return rows
    
    def compare_catalogues_componentwise(self, gt_rows: list[dict], pred_rows: list[dict]) -> dict:
        """
        Compare GT and predicted catalogues at the component level, with classwise breakdown.

        TP: component detected under correct source
        FN: component not detected, or detected under wrong source
        FP: never happens
        Precision is always 1.0, recall = TP / (TP + FN) = F1 = recall
        """
        gt_comp_to_source = {r["comp_name"]: r["source_name"] for r in gt_rows}
        gt_comp_to_class = {r["comp_name"]: r["class"] for r in gt_rows}

        pred_comp_to_source = {}
        pred_comp_to_class = {}
        pred_comp_best_score = {}

        for r in pred_rows:
            comp = r["comp_name"]
            score = r["pred_score"]
            if comp not in pred_comp_best_score or score > pred_comp_best_score[comp]:
                pred_comp_best_score[comp] = score
                pred_comp_to_source[comp] = r["source_name"]
                pred_comp_to_class[comp] = r["class"]

        results = []

        for comp_name, gt_source in gt_comp_to_source.items():
            gt_class = gt_comp_to_class[comp_name]
            pred_source = pred_comp_to_source.get(comp_name)
            pred_class = pred_comp_to_class.get(comp_name)

            if pred_source is None:
                results.append({
                    "comp_name": comp_name,
                    "gt_source": gt_source,
                    "pred_source": None,
                    "gt_class": gt_class,
                    "pred_class": None,
                    "status": "FN",
                    "reason": "not detected",
                })
            elif pred_source == gt_source:
                results.append({
                    "comp_name": comp_name,
                    "gt_source": gt_source,
                    "pred_source": pred_source,
                    "gt_class": gt_class,
                    "pred_class": pred_class,
                    "status": "TP",
                    "reason": None,
                })
            else:
                results.append({
                    "comp_name": comp_name,
                    "gt_source": gt_source,
                    "pred_source": pred_source,
                    "gt_class": gt_class,
                    "pred_class": pred_class,
                    "status": "FN",
                    "reason": f"assigned to wrong source: {pred_source}",
                })

        # Only FP for components predicted that don't exist in GT at all
        # for comp_name, pred_source in pred_comp_to_source.items():
        #     if comp_name not in gt_comp_to_source:
        #         results.append({
        #             "comp_name": comp_name,
        #             "gt_source": None,
        #             "pred_source": pred_source,
        #             "gt_class": None,
        #             "pred_class": pred_comp_to_class[comp_name],
        #             "status": "FP",
        #             "reason": "hallucinated component",
        #         })

        # # Wrong-source detections also generate a FP entry
        # for r in list(results):
        #     if r["status"] == "FN" and r["reason"] and "wrong source" in r["reason"]:
        #         results.append({
        #             "comp_name": r["comp_name"],
        #             "gt_source": None,
        #             "pred_source": r["pred_source"],
        #             "gt_class": None,
        #             "pred_class": r["pred_class"],
        #             "status": "FP",
        #             "reason": f"incorrectly assigned to {r['pred_source']} instead of {r['gt_source']}",
        #         })

        summary = self._compute_summary_classwise(results)
        return {"results": results, "summary": summary}


    def compare_catalogues(self, gt_rows: list[dict], pred_rows: list[dict]) -> dict:
        """
        Compare GT and predicted catalogues at the source level, with classwise breakdown.
        """
        gt_by_source = self._group_by_source(gt_rows)
        pred_by_source = self._group_by_source(pred_rows)

        results = []

        for source_name, gt_comps in gt_by_source.items():
            gt_comp_names = set(r["comp_name"] for r in gt_comps)
            gt_class = gt_comps[0]["class"]
            pred_comps = pred_by_source.get(source_name)

            if pred_comps is None:
                results.append({
                    "source_name": source_name,
                    "gt_class": gt_class,
                    "pred_class": None,
                    "gt_components": gt_comp_names,
                    "pred_components": set(),
                    "status": "FN",
                    "missing_components": gt_comp_names,
                    "extra_components": set(),
                })
                continue

            pred_comp_names = set(r["comp_name"] for r in pred_comps)
            pred_class = pred_comps[0]["class"]
            missing = gt_comp_names - pred_comp_names
            extra = pred_comp_names - gt_comp_names
            status = "TP" if not missing and not extra else "FP"

            results.append({
                "source_name": source_name,
                "gt_class": gt_class,
                "pred_class": pred_class,
                "gt_components": gt_comp_names,
                "pred_components": pred_comp_names,
                "status": status,
                "missing_components": missing,
                "extra_components": extra,
            })

        for source_name, pred_comps in pred_by_source.items():
            if source_name not in gt_by_source:
                results.append({
                    "source_name": source_name,
                    "gt_class": None,
                    "pred_class": pred_comps[0]["class"],
                    "gt_components": set(),
                    "pred_components": set(r["comp_name"] for r in pred_comps),
                    "status": "FP",
                    "missing_components": set(),
                    "extra_components": set(r["comp_name"] for r in pred_comps),
                })

        summary = self._compute_summary_classwise(results)
        return {"results": results, "summary": summary}


    def _compute_summary_classwise(self, results: list[dict]) -> dict:
        """
        Compute overall and per-class TP/FP/FN/precision/recall/F1.
        Class is taken from gt_class for TP/FN, pred_class for FP.
        """
        classes = set()
        for r in results:
            if r["gt_class"]:
                classes.add(r["gt_class"])
            if r["pred_class"]:
                classes.add(r["pred_class"])

        def _metrics(tp, fp, fn):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1}

        # Overall counts
        overall_tp = sum(1 for r in results if r["status"] == "TP")
        overall_fp = sum(1 for r in results if r["status"] == "FP")
        overall_fn = sum(1 for r in results if r["status"] == "FN")

        classwise = {}
        for cls in classes:
            # TP: correctly detected, gt_class matches
            tp = sum(1 for r in results if r["status"] == "TP" and r["gt_class"] == cls)
            # FN: missed, gt_class matches
            fn = sum(1 for r in results if r["status"] == "FN" and r["gt_class"] == cls)
            # FP: false alarm attributed to this class via pred_class
            fp = sum(1 for r in results if r["status"] == "FP" and r["pred_class"] == cls)
            classwise[cls] = _metrics(tp, fp, fn)

        return {
            "overall": _metrics(overall_tp, overall_fp, overall_fn),
            "classwise": classwise,
        }


    def _compute_summary(self, results: list[dict]) -> dict:
        """Keep for backwards compatibility — delegates to classwise."""
        return self._compute_summary_classwise(results)

    def _group_by_source(self, rows: list[dict]) -> dict:
        """Group catalogue rows by source_name."""
        grouped = {}
        for row in rows:
            source_name = row["source_name"]
            if source_name not in grouped:
                grouped[source_name] = []
            grouped[source_name].append(row)
        return grouped
    
    def _build_pred_catalogue_with_masks(self, image_id: int, pred_idx: int, predictions: PredictionObject, pos_map: dict):
        mask = predictions.get_pred_masks()
        pred_class = predictions.get_pred_classes()
        pred_score = predictions.get_pred_scores()

        # Check which positions fall inside the mask
        rows = []
        for pos, comp_info in pos_map.items():
            x, y = pos
            if mask[y, x]:  # Assuming mask is a 2D array where True indicates the predicted area
                rows.append({
                    "comp_name": comp_info["component_name"],
                    "source_name": comp_info["source_name"],
                    "class": pred_class,
                    "pred_idx": pred_idx,
                    "pred_score": pred_score,
                    "image_id": image_id,
                })
        rows = self._resolve_duplicate_predictions(rows)
        return rows
    
    def _build_pred_catalogue_with_boxes(self, image_id: int, pred_idx: int, predictions: PredictionObject, pos_map: dict):
        x1, y1, x2, y2 = predictions.get_pred_boxes()
        pred_class = predictions.get_pred_classes()
        pred_score = predictions.get_pred_scores()

        # Check which positions fall inside the bounding box
        rows = []
        for pos, comp_info in pos_map.items():
            x, y = pos
            if x1 <= x <= x2 and y1 <= y <= y2:
                rows.append({
                    "comp_name": comp_info["component_name"],
                    "source_name": comp_info["source_name"],
                    "class": pred_class,
                    "pred_idx": pred_idx,
                    "pred_score": pred_score,
                    "image_id": image_id,
                })
        rows = self._resolve_duplicate_predictions(rows)
        return rows
    
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
    