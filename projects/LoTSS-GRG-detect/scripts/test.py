#!/usr/bin/env python
"""
Testing script for GRG detection using Mask R-CNN with precomputed proposals.

This script mirrors the notebook evaluation flow (without plotting):
- setup imports/configuration
- register datasets from dataset config
- load model weights/checkpoint
- run evaluation with COCOEvaluator + GRGEvaluator
- save results JSON
- print a human-readable summary

Usage examples:
	python test.py
	python test.py --weights /path/to/model_0052499.pth
	python test.py --config-file ../configs/mask_rcnn_R_50_FPN_grg.yaml \
				   --dataset-config ../config/dataset.yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add detectron2 to path (assumes detectron2 is in the parent directory structure)
detectron2_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(detectron2_root))

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset

from data.dataset_mapper import GRGDatasetMapper
from data.register_dataset import main as register_datasets
from evaluation.grg_evaluator import GRGEvaluator


logger = logging.getLogger("LoTSS-GRG-detect.test")


def _find_test_dataset_name(registered_datasets):
	"""Find the test dataset name from the registered dataset list."""
	if not registered_datasets:
		raise ValueError("No datasets were registered from dataset config.")

	candidates = [d for d in registered_datasets if "test" in d.lower()]
	if len(candidates) == 1:
		return candidates[0]
	if len(candidates) > 1:
		return candidates[0]

	return registered_datasets[-1]


def _serialize_results(results):
	"""Convert detectron2 results dict into JSON-serializable structure."""
	serializable_results = {}
	for key, value in results.items():
		if isinstance(value, dict):
			serializable_results[key] = {
				metric: float(metric_value)
				if isinstance(metric_value, (int, float, np.number))
				else str(metric_value)
				for metric, metric_value in value.items()
			}
		else:
			serializable_results[key] = value
	return serializable_results


def _print_summary(results, model_weights, test_dataset_name, num_test_images, score_threshold, output_dir):
	"""Print summary similar to notebook cell 9."""
	print("=" * 80)
	print("TEST SET EVALUATION SUMMARY")
	print("=" * 80)
	print(f"\nModel: {model_weights}")
	print(f"Test Dataset: {test_dataset_name} ({num_test_images} images)")
	print(f"Score Threshold: {score_threshold}")
	print("\n" + "=" * 80)

	if "GRG" in results:
		print("\nGRG Detection Metrics:")
		print(f"  Segmentation Accuracy:  {results['GRG']['segm_accuracy']:.1%}")
		print(f"  Segmentation Precision: {results['GRG']['segm_precision']:.1%}")
		print(f"  Segmentation Recall:    {results['GRG']['segm_recall']:.1%}")
		print(f"  Segmentation F1 Score:  {results['GRG']['segm_f1']:.1%}")
		print(f"  Bounding Box Accuracy:  {results['GRG']['bbox_accuracy']:.1%}")
		print(f"  Bounding Box Precision: {results['GRG']['bbox_precision']:.1%}")
		print(f"  Bounding Box Recall:    {results['GRG']['bbox_recall']:.1%}")
		print(f"  Bounding Box F1 Score:  {results['GRG']['bbox_f1']:.1%}")

	if "segm" in results:
		print("\nCOCO Segmentation Metrics:")
		for key in ["AP", "AP50", "AP75", "APs", "APm", "APl"]:
			if key in results["segm"]:
				print(f"  {key}: {results['segm'][key]:.1f}")

	if "bbox" in results:
		print("\nCOCO Detection Metrics:")
		for key in ["AP", "AP50", "AP75", "APs", "APm", "APl"]:
			if key in results["bbox"]:
				print(f"  {key}: {results['bbox'][key]:.1f}")

	print("\n" + "=" * 80)
	print(f"\nAll results saved to: {output_dir}")
	print("=" * 80)


def parse_args():
	parser = argparse.ArgumentParser(description="Test GRG model with COCO + custom GRG evaluation")
	parser.add_argument(
		"--config-file",
		default=str(project_root / "configs/mask_rcnn_R_50_FPN_grg.yaml"),
		help="Path to model config YAML",
	)
	parser.add_argument(
		"--dataset-config",
		default=str(project_root / "config/dataset.yaml"),
		help="Path to dataset config YAML",
	)
	parser.add_argument(
		"--weights",
		default=None,
		help="Path to model checkpoint (.pth). Overrides MODEL.WEIGHTS from config",
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="Directory where evaluation outputs/results are written",
	)
	parser.add_argument(
		"--score-threshold",
		type=float,
		default=None,
		help="Score threshold for inference and GRG evaluator",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	print(f"PyTorch version: {torch.__version__}")
	print(f"CUDA available: {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		print(f"CUDA version: {torch.version.cuda}")
		print(f"GPU: {torch.cuda.get_device_name(0)}")

	config_file = os.path.abspath(args.config_file)
	dataset_config = os.path.abspath(args.dataset_config)

	logger.info(f"Model config: {config_file}")
	logger.info(f"Dataset config: {dataset_config}")

	if not os.path.exists(config_file):
		raise FileNotFoundError(f"Model config not found: {config_file}")
	if not os.path.exists(dataset_config):
		raise FileNotFoundError(f"Dataset config not found: {dataset_config}")

	registered = register_datasets(dataset_config)
	logger.info(f"Registered datasets: {registered}")

	test_dataset_name = _find_test_dataset_name(registered)
	logger.info(f"Using test dataset: {test_dataset_name}")

	test_data = DatasetCatalog.get(test_dataset_name)
	logger.info(f"Number of test images: {len(test_data)}")
	if len(test_data) == 0:
		raise RuntimeError(f"Dataset {test_dataset_name} is empty.")

	cfg = get_cfg()
	cfg.merge_from_file(config_file)

	if args.weights:
		cfg.MODEL.WEIGHTS = os.path.abspath(args.weights)

	if args.score_threshold is not None:
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold

	cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	cfg.DATASETS.TEST = (test_dataset_name,)

	if args.output_dir:
		cfg.OUTPUT_DIR = os.path.abspath(args.output_dir)

	output_dir = os.path.abspath(cfg.OUTPUT_DIR)
	os.makedirs(output_dir, exist_ok=True)

	logger.info(f"Model weights: {cfg.MODEL.WEIGHTS}")
	logger.info(f"Output directory: {output_dir}")
	logger.info(f"Device: {cfg.MODEL.DEVICE}")
	logger.info(f"Score threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")

	if not os.path.exists(cfg.MODEL.WEIGHTS):
		raise FileNotFoundError(
			f"Model checkpoint not found: {cfg.MODEL.WEIGHTS}. "
			"Provide a valid checkpoint with --weights or in the model config."
		)

	predictor = DefaultPredictor(cfg)
	logger.info("Model loaded successfully")

	annotations_path = MetadataCatalog.get(test_dataset_name).json_file

	evaluators = DatasetEvaluators(
		[
			COCOEvaluator(test_dataset_name, output_dir=output_dir),
			GRGEvaluator(
				coco_images=test_data,
				annotations_path=annotations_path,
				score_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
			),
		]
	)

	test_loader = build_detection_test_loader(
		cfg,
		test_dataset_name,
		mapper=GRGDatasetMapper(cfg, is_train=False),
	)

	print(f"Running evaluation on {len(test_data)} test images...")

	results = inference_on_dataset(predictor.model, test_loader, evaluators)

	print("\n" + "=" * 80)
	print("EVALUATION RESULTS")
	print("=" * 80)
	for task, metrics in results.items():
		print(f"\n{task.upper()}:")
		if isinstance(metrics, dict):
			for metric_name, value in metrics.items():
				if isinstance(value, float):
					print(f"  {metric_name}: {value:.4f}")
				else:
					print(f"  {metric_name}: {value}")
		else:
			print(f"  {metrics}")

	results_file = os.path.join(output_dir, "test_results.json")
	with open(results_file, "w") as f:
		json.dump(_serialize_results(results), f, indent=2)
	print(f"\n✓ Results saved to: {results_file}")

	_print_summary(
		results=results,
		model_weights=cfg.MODEL.WEIGHTS,
		test_dataset_name=test_dataset_name,
		num_test_images=len(test_data),
		score_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
		output_dir=output_dir,
	)


if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	)
	main()
