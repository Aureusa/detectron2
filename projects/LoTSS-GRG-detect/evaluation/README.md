# Evaluation — Overview

This folder contains two evaluators for Giant Radio Galaxy (GRG) detection, both inheriting from the base `GRGEvaluator`. Each one asks a slightly different question about how well the model performs.

---

## Shared foundation

Both evaluators share:

- A COCO annotations file as ground truth source.
- A `score_threshold` to filter out low-confidence predictions (default `0.5`).
- The standard Detectron2 `process()` / `evaluate()` / `reset()` interface.
- Helper metrics: `_precision`, `_recall`, `_f1`, `_accuracy`.

The COCO annotations must include, per image, a `metadata` field with a `candidates.grouping.components` list. Each component entry carries:

```json
{
  "component_name": "ILTJ123456.7+123456",
  "source_name":    "GRG-0042",
  "source_id":      42,
  "xy":             [134, 201]
}
```

This pixel position `xy` is the central coordinate of the radio component and is the key that connects a predicted spatial region to an astronomical source.

---

## 1. `B2SMaskedRCNNEvaluator`

Measures whether the model produces spatially exact instance detections.

### How it works

For each image, predictions are filtered by score threshold; all above-threshold instances are kept. Ground-truth is loaded directly from COCO annotations via `instance_positions` — the list of pixel coordinates belonging to each annotated instance.

Matching is done per image with a greedy one-to-one assignment:

1. For each GT annotation, iterate over every unmatched prediction of the same class.
2. Check **containment**: does the predicted mask (or bbox) cover every position in `instance_positions`?
3. Check **exclusivity**: does the prediction bleed into any position that belongs to a *different* annotation in the image?
4. Among all valid candidate predictions, pick the one with the highest score and mark both as matched.

After matching:

| Count | Meaning |
|---|---|
| **TP** | Matched GT annotations |
| **FN** | Unmatched GT annotations |
| **FP** | Unmatched predictions |

This is computed separately for **mask** containment and **bbox** containment, yielding `segm_*` and `bbox_*` metrics.

### TP/FP/FN definitions

- **TP**: a prediction covers *all* pixel positions of a GT annotation and *no* pixel positions of any other annotation in the image, and the predicted class matches.
- **FP**: a valid prediction exists but is unmatched (wrong class, partial coverage, or bleeds into another annotation).
- **FN**: a GT annotation has no matching prediction.

### Output keys (`"B2S"`)

```
segm_precision, segm_recall, segm_f1, segm_accuracy
bbox_precision, bbox_recall, bbox_f1, bbox_accuracy
```

Plus per-class variants under `"B2S_CLASSWISE"` (e.g. `SCS_segm_f1`, `MCS_bbox_recall`).

---

## 2. `ComponentAssociationEvaluator`

Measures whether the model correctly groups radio components into their astronomical sources, without requiring exact spatial boundaries.

### Catalogue construction

For every image, `CatalogueConstructor` (in `engine/cat_constructor.py`) builds two dictionaries mapping source names to their component sets.

**GT catalogue** (`build_gt_catalogue`):
Reads each COCO annotation and maps its `instance_positions` to component names via the `xy` position map. Each annotation produces one entry:

```python
"GRG-0042": {
    "components": {"ILTJ123456.7+123456", "ILTJ123456.7+654321"},
    "class": "MCS",
    "score": None,
}
```

**Prediction catalogue** (`build_pred_catalogue`):
For each above-threshold prediction, scans every known component pixel position in the image. Any position that falls inside the predicted mask or bbox is added to that prediction's component set. Each prediction becomes a blindly-named entry:

```python
"pred_source_0": {
    "components": {"ILTJ123456.7+123456", "ILTJ123456.7+654321"},
    "class": "MCS",
    "score": 0.92,
}
```

The prediction is **never told the source name** — it only knows which component coordinates it spatially covers. The model implicitly groups components by covering their pixel positions.

### Matching via the Hungarian algorithm

After building both catalogues, `compare_catalogues` computes an N×M similarity matrix between all GT sources and all predicted sources using **Jaccard similarity** (IoU) of their component sets:

$$\text{similarity}(g, p) = \frac{|G_g \cap P_p|}{|G_g \cup P_p|}$$

`scipy.optimize.linear_sum_assignment` then finds the optimal 1-to-1 assignment that maximises total similarity. Each matched (GT, pred) pair is evaluated as follows:

| Condition | Status |
|---|---|
| similarity = 1.0 and class matches | **TP** — perfect match |
| 0 < similarity < 1.0 and class matches | **FP** — partial match (extra or missing components) |
| similarity > 0.0 and class mismatch | **FP** — wrong class |
| similarity = 0.0 | **FN** — no overlap at all |

After the assignment loop, any unmatched GT source is a **missed source (FN)** and any unmatched prediction is a **hallucinated source (FP)**.

### Evaluation levels

Both source-level and component-level TP/FP/FN are accumulated simultaneously during the sourcewise comparison pass:

**Source-level**: one TP/FP/FN per source, as defined by the table above.

**Component-level**: from each matched (GT, pred) pair the intersection, missed, and extra components are counted directly:
- `tp_component` = |GT ∩ Pred|
- `fn_component` = |GT − Pred| (missed components)
- `fp_component` = |Pred − GT| (hallucinated components)

Unmatched GT sources contribute all their components to `fn_component`; unmatched predictions contribute all their components to `fp_component`.

### Output keys (`"CAE"`)

The return dict contains six sub-dicts, one per (split × modality) combination:

```
mask_overall_results, mask_scs_results, mask_mcs_results
bbox_overall_results, bbox_scs_results, bbox_mcs_results
```

Each sub-dict contains:

```
{prefix}_source_{split}_precision / recall / f1 / accuracy   (source-level)
{prefix}_component_{split}_precision / recall / f1 / accuracy (component-level)
{prefix}_{split}_tp_source, fp_source, fn_source
{prefix}_{split}_tp_component, fp_component, fn_component
{prefix}_{split}_perfect_match, partial_match, wrong_class, no_match,
         missed_source, hallucinated_source
```

where `prefix` is `mask` or `bbox` and `split` is `overall`, `scs`, or `mcs`.

---

## Comparison

| Aspect | `B2SMaskedRCNNEvaluator` | `ComponentAssociationEvaluator` |
|---|---|---|
| **Core question** | Did you produce a spatially exact detection of this annotated instance? | Did you correctly group these radio components into their source? |
| **Matching** | Greedy 1-to-1, class-aware, by spatial containment | Optimal 1-to-1 via Hungarian algorithm on Jaccard similarity of component sets |
| **TP criterion** | Prediction contains ALL positions of a GT annotation AND no positions of any other annotation | Component-set IoU = 1.0 and class matches |
| **Spatial exclusivity** | Enforced at the pixel level — any bleed into a foreign annotation's coordinates fails | Not enforced — only named component coordinates count; background bleed is ignored |
| **Multiple predictions** | All above-threshold predictions compete; each GT can match at most one | All predictions enter the similarity matrix; Hungarian algorithm finds the global optimum |
| **Partial credit** | None — a missed component means no TP at all | Yes — component-level metrics give credit per correctly grouped component |
| **FP semantics** | Unmatched predictions, or predictions that grab another annotation's pixels | Partial component match, wrong class, or hallucinated grouping with no GT overlap |
| **Strictness** | Very strict: exact spatial boundary, class match, exclusive occupancy | Source-level strict (IoU must be 1.0); component-level lenient (partial overlap accumulates TP) |

### When to use which

- **`B2SMaskedRCNNEvaluator`** is the right metric when you want to know whether the model's segmentation masks are accurate at the pixel level. It mirrors standard instance segmentation evaluation but replaces IoU with a domain-specific containment and exclusivity check.

- **`ComponentAssociationEvaluator`** is the right metric for the scientific downstream task: given the model's predicted regions, did it correctly associate multi-component radio sources? It is more forgiving of imprecise spatial boundaries (background bleed does not penalise) and rewards correct grouping. The component-level sub-metric additionally shows how many individual components were correctly recovered regardless of perfect source boundary.

---

## Caveats

1. **MCS recall is easier to achieve in CAE than in B2S**: for a multi-component source, B2S requires the predicted region to cover all component coordinates *and* not sweep over any other annotation's coordinates — extremely hard for spatially extended sources. CAE only requires the right discrete component names in the set, so background bleed between components is not penalised.

2. **Source name is never used by the model**: in `ComponentAssociationEvaluator`, the predicted catalogue uses opaque names (`pred_source_N`). Source attribution is determined entirely by which component coordinates fall spatially inside the prediction. The GT source name is only used on the GT side for the similarity matrix.

3. **Class label applied per prediction**: `build_pred_catalogue` stamps all components covered by a prediction with that prediction's class. A misclassified MCS source will have all its components attributed to the wrong class, distorting per-class component metrics.

4. **Score threshold affects both evaluators**: lowering the threshold admits more predictions, which can increase recall at the cost of precision. `ComponentAssociationEvaluator` exposes `set_score_threshold()` to allow post-hoc threshold sweeps without re-running inference.
