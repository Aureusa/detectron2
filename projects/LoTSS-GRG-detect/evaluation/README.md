# GRG Evaluator for Detectron2

## Overview

The `GRGEvaluator` implements a custom evaluator for Giant Radio Galaxy (GRG) detection that follows the detectron2 evaluation framework.

## Usage

### 1. Basic Integration

```python
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader
from evaluation.grg_evaluator import GRGEvaluator

# Initialize the evaluator
evaluator = GRGEvaluator(
    annotations=dataset_dict,  # Your COCO-format annotations dict
    annotations_path="path/to/annotations.json",
    score_threshold=0.5  # Confidence threshold for predictions
)

# Run evaluation
results = inference_on_dataset(model, data_loader, evaluator)
print(results)
```

### 2. Integration in train_net.py

```python
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import DatasetEvaluator
from evaluation.grg_evaluator import GRGEvaluator

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        # Load your annotations
        from detectron2.data import DatasetCatalog, MetadataCatalog
        import json
        
        # Get the annotations path from metadata
        metadata = MetadataCatalog.get(dataset_name)
        annotations_path = metadata.json_file
        
        # Load annotations dict
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        return GRGEvaluator(
            annotations=annotations,
            annotations_path=annotations_path,
            score_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        )
```

### 3. Required Data Format

Your COCO annotations JSON should include metadata for each image:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "height": 512,
      "width": 512,
      "metadata": {
        "grg_positions": [[100, 150], [200, 250]],
        "all_component_positions": [[100, 150], [200, 250], [300, 350]]
      }
    }
  ],
  "annotations": [...],
  "categories": [...]
}
```

## Metrics

The evaluator computes three metrics for GRG detection:

- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)

### Definitions

- **True Positive (TP)**: Predicted region uniquely encompasses all GRG components and no non-GRG components
- **False Positive (FP)**: Predicted region is missing GRG components OR includes non-GRG components
- **False Negative (FN)**: No predicted region covering GRG components (above threshold)

## Configuration

The evaluator accepts the following parameters:

- `annotations` (dict): Complete COCO-format annotations dictionary
- `annotations_path` (str): Path to the annotations JSON file (for pycocotools)
- `score_threshold` (float, default=0.5): Minimum confidence score for considering predictions

## Output Format

Results are returned in detectron2's expected format:

```python
{
    "grg": {
        "accuracy": 0.85,
        "precision": 0.90,
        "recall": 0.88
    }
}
```

## Notes

- The evaluator works with both instance segmentation (masks) and object detection (boxes)
- For detection-only models, bounding boxes are converted to binary masks
- All predictions below `score_threshold` are ignored
- The evaluator handles distributed evaluation automatically when used with `inference_on_dataset`
