#!/usr/bin/env python
"""
Testing script for RaCUN.

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
import json
import logging
import os
import sys
from pathlib import Path
import datetime
import logging
import time
from contextlib import ExitStack
import torch
from torch import nn

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
from detectron2.engine import DefaultPredictor, default_setup, default_argument_parser
from detectron2.evaluation import inference_context
from detectron2.utils.logger import setup_logger, log_every_n_seconds
from detectron2.utils.comm import get_world_size

from data.dataset_mapper import GRGDatasetMapper
from data.register_dataset import main as register_datasets
from evaluation.grg_evaluator import B2SMaskedRCNNEvaluator
from configs.custom_config import add_lotss_grg_config

logger = setup_logger(name="LoTSS-GRG-detect.test_masked_rcnn", termcolor="magenta")


def setup(args):
    """
    Create config and perform basic setups.
    """
    cfg = get_cfg()
    add_lotss_grg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.dataset_config:
        dataset_config_path = os.path.normpath(args.dataset_config)
    else:
        raise ValueError("Dataset config file must be provided with --dataset-config")
    # Ensure output directory exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    default_setup(cfg, args)
    
    # Register datasets   
    logger.info(f"Registering datasets from: {dataset_config_path}")
    registered = register_datasets(dataset_config_path)
    logger.info(f"Registered datasets: {registered}")
    
    # Add proposal directories to metadata
    for dataset_name in registered:
        split = dataset_name.split("_")[-1]  # train, val, or test
        proposal_dir = os.path.join(
            cfg.DATASETS.get("DATA_ROOT", "/home/s4861264/project_data/full-dataset"),
            split,
            "proposals"
        )
        MetadataCatalog.get(dataset_name).set(proposal_dir=proposal_dir)
        logger.info(f"Set proposal_dir for {dataset_name}: {proposal_dir}")
    
    return cfg

def process_with_evaluator(
    model,
    data_loader,
    evaluator: B2SMaskedRCNNEvaluator,
    callbacks=None,
):
	"""
	Run model on the data_loader and evaluate the metrics with evaluator.
	Also benchmark the inference speed of `model.__call__` accurately.
	The model will be used in eval mode.

	Args:
		model (callable): a callable which takes an object from
			`data_loader` and returns some outputs.

			If it's an nn.Module, it will be temporarily set to `eval` mode.
			If you wish to evaluate a model in `training` mode instead, you can
			wrap the given model and override its behavior of `.eval()` and `.train()`.
		data_loader: an iterable object with a length.
			The elements it generates will be the inputs to the model.
		evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
			but don't want to do any evaluation.
		callbacks (dict of callables): a dictionary of callback functions which can be
			called at each stage of inference.

	Returns:
		The return value of `evaluator.evaluate()`
	"""
	num_devices = get_world_size()
	logger = logging.getLogger(__name__)
	logger.info("Start inference on {} batches".format(len(data_loader)))

	total = len(data_loader)  # inference data loader must have a fixed length

	if evaluator is None:
		raise ValueError("Evaluator must be provided for evaluation.")

	evaluator.reset()

	num_warmup = min(5, total - 1)
	start_time = time.perf_counter()
	total_data_time = 0
	total_compute_time = 0
	total_eval_time = 0
	with ExitStack() as stack:
		if isinstance(model, nn.Module):
			stack.enter_context(inference_context(model))
		stack.enter_context(torch.no_grad())

		start_data_time = time.perf_counter()
		dict.get(callbacks or {}, "on_start", lambda: None)()
		for idx, inputs in enumerate(data_loader):
			total_data_time += time.perf_counter() - start_data_time
			if idx == num_warmup:
				start_time = time.perf_counter()
				total_data_time = 0
				total_compute_time = 0
				total_eval_time = 0

			start_compute_time = time.perf_counter()
			dict.get(callbacks or {}, "before_inference", lambda: None)()
			outputs = model(inputs)
			dict.get(callbacks or {}, "after_inference", lambda: None)()
			if torch.cuda.is_available():
				torch.cuda.synchronize()
			total_compute_time += time.perf_counter() - start_compute_time

			start_eval_time = time.perf_counter()
			evaluator.process(inputs, outputs)
			total_eval_time += time.perf_counter() - start_eval_time

			iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
			data_seconds_per_iter = total_data_time / iters_after_start
			compute_seconds_per_iter = total_compute_time / iters_after_start
			eval_seconds_per_iter = total_eval_time / iters_after_start
			total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
			if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
				eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
				log_every_n_seconds(
					logging.INFO,
					(
						f"Inference done {idx + 1}/{total}. "
						f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
						f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
						f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
						f"Total: {total_seconds_per_iter:.4f} s/iter. "
						f"ETA={eta}"
					),
					n=5,
				)
			start_data_time = time.perf_counter()
		dict.get(callbacks or {}, "on_end", lambda: None)()

	# Measure the time only for this worker (before the synchronization barrier)
	total_time = time.perf_counter() - start_time
	total_time_str = str(datetime.timedelta(seconds=total_time))
	# NOTE this format is parsed by grep
	logger.info(
		"Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
			total_time_str, total_time / (total - num_warmup), num_devices
		)
	)
	total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
	logger.info(
		"Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
			total_compute_time_str,
			total_compute_time / (total - num_warmup),
			num_devices,
		)
	)
	return evaluator


def _find_test_dataset_name(registered_datasets, dataset_type="val"):
	"""Find the test dataset name from the registered dataset list."""
	if not registered_datasets:
		raise ValueError("No datasets were registered from dataset config.")

	candidates = [d for d in registered_datasets if dataset_type in d.lower()]
	if len(candidates) == 1:
		return candidates[0]
	if len(candidates) > 1:
		return candidates[0]

	return registered_datasets[-1]


def _serialize_results(results):
	"""Convert detectron2 results dict into JSON-serializable structure."""
	def _to_jsonable(value):
		if isinstance(value, dict):
			# Evaluator returns {"B2S": {...}}; unwrap it so results.json stores metrics directly.
			if set(value.keys()) == {"B2S"} and isinstance(value["B2S"], dict):
				return _to_jsonable(value["B2S"])
			return {metric: _to_jsonable(metric_value) for metric, metric_value in value.items()}
		if isinstance(value, (list, tuple)):
			return [_to_jsonable(item) for item in value]
		if isinstance(value, np.generic):
			return value.item()
		if isinstance(value, torch.Tensor):
			if value.numel() == 1:
				return value.item()
			return value.detach().cpu().tolist()
		return value

	return {key: _to_jsonable(value) for key, value in results.items()}


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
	parser = default_argument_parser()
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
		"--dataset-type",
		default="val",
		choices=["train", "val", "test"],
		help="Which dataset split to evaluate on (train/val/test). Default is 'val'.",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	cfg = setup(args)

	output_dir = args.output_dir if args.output_dir else cfg.OUTPUT_DIR
	os.makedirs(output_dir, exist_ok=True)

	config_file = os.path.abspath(args.config_file)
	dataset_config = os.path.abspath(args.dataset_config)

	# Get the registered test dataset name (e.g. "lotss_grg_test") based on the dataset type specified in args
	all_registered_dataset = DatasetCatalog.list()
	dataset_name = "b2s" # i.e. "b2s"
	registered_dataset = [d for d in all_registered_dataset if dataset_name in d]
	test_dataset_name = _find_test_dataset_name(registered_dataset, args.dataset_type)
	logger.info(f"Using `{test_dataset_name}` dataset for testing.")

	if args.weights:
		cfg.MODEL.WEIGHTS = os.path.abspath(args.weights)
	cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	cfg.DATASETS.TEST = (test_dataset_name,)
	cfg.OUTPUT_DIR = os.path.abspath(output_dir)

	if not os.path.exists(cfg.MODEL.WEIGHTS):
		raise FileNotFoundError(
			f"Model checkpoint not found: {cfg.MODEL.WEIGHTS}. "
			"Provide a valid checkpoint with --weights or in the model config."
		)

	# Log configuration
	training_info = "Testing configuration:"
	training_info += f"\n  -Model: Mask R-CNN with ResNet-50 FPN"
	training_info += f"\n  -Training dataset: {cfg.DATASETS.TRAIN}"
	training_info += f"\n  -Validation dataset: {cfg.DATASETS.TEST}"
	training_info += f"\n  -Test dataset: {cfg.DATASETS.TEST}"
	training_info += f"\n  -Results directory: {output_dir}"
	training_info += f"\n  -Model config: {config_file}"
	training_info += f"\n  -Dataset config: {dataset_config}"
	training_info += f"\n  -Model weights: {cfg.MODEL.WEIGHTS}"
	training_info += f"\n  -Output directory: {output_dir}"
	training_info += f"\n  -Device: {cfg.MODEL.DEVICE}"
	logger.info(training_info)

	system_info = "System Information:"
	system_info += f"\n  -PyTorch version: {torch.__version__}"
	system_info += f"\n  -CUDA available: {torch.cuda.is_available()}"
	if torch.cuda.is_available():
		system_info += f"\n  -CUDA version: {torch.version.cuda}"
		system_info += f"\n  -GPU: {torch.cuda.get_device_name(0)}"
	logger.info(system_info)

	if not os.path.exists(config_file):
		raise FileNotFoundError(f"Model config not found: {config_file}")
	if not os.path.exists(dataset_config):
		raise FileNotFoundError(f"Dataset config not found: {dataset_config}")

	test_data = DatasetCatalog.get(test_dataset_name)
	logger.info(f"Number of test images: {len(test_data)}")
	if len(test_data) == 0:
		raise RuntimeError(f"Dataset {test_dataset_name} is empty.")

	predictor = DefaultPredictor(cfg)
	logger.info("Model loaded successfully")

	# Get annotations path from metadata (set by register_coco_instances)
	metadata = MetadataCatalog.get(test_dataset_name)
	if not hasattr(metadata, "json_file"):
		raise AttributeError(
			f"Attribute 'json_file' does not exist in metadata for dataset '{test_dataset_name}'. "
			"Ensure dataset registration uses register_coco_instances and that the dataset config "
			"points to a valid COCO-style annotations file."
		)
	annotations_path = metadata.json_file

	# Thresholds for Membership Head - 11 thresholds from 0.0 to 1.0 inclusive
	thresholds = np.arange(0.0, 1.1, 0.1)
	evaluator = B2SMaskedRCNNEvaluator(
		coco_images=DatasetCatalog.get(test_dataset_name),
		annotations_path=annotations_path,
		score_threshold=thresholds[0],  # Initial threshold; will be updated in loop
	)

	test_loader = build_detection_test_loader(
		cfg,
		test_dataset_name,
		mapper=GRGDatasetMapper(cfg, is_train=False),
	)

	logger.info(
		"Running inference on {} batches ({} images) with {} devices...".format(
			len(test_loader),
			len(test_data),
			torch.cuda.device_count()
		)
	)
	evaluator = process_with_evaluator(
		predictor.model,
		test_loader,
		evaluator,
		callbacks=None,
	)

	best_scs_assoc_f1 = -1
	best_mcs_assoc_f1 = -1
	best_scs_thresh = None
	best_mcs_thresh = None
	results = {}
	for t in thresholds:
		evaluator.set_score_threshold(t)
		current_results = evaluator.evaluate()
		results[f"threshold_{t:.1f}"] = current_results["B2S"]
		results[f"threshold_{t:.1f}_classwise"] = current_results["B2S_CLASSWISE"]

		curr_scs_assoc_f1 = current_results["B2S_CLASSWISE"]["SCS_segm_f1"]
		curr_mcs_assoc_f1 = current_results["B2S_CLASSWISE"]["MCS_segm_f1"]

		if curr_scs_assoc_f1 > best_scs_assoc_f1:
			best_scs_assoc_f1 = curr_scs_assoc_f1
			best_scs_thresh = t  # Membership and Validity thresholds are the same in this setup

		if curr_mcs_assoc_f1 > best_mcs_assoc_f1:
			best_mcs_assoc_f1 = curr_mcs_assoc_f1
			best_mcs_thresh = t

	serialized_results = _serialize_results(results)

	logger.info(
		f"Best SCS Association F1: {best_scs_assoc_f1:.1%} "
		f"at Membership Threshold = {best_scs_thresh:.1f}, "
		f"Validity Threshold = {best_scs_thresh:.1f}"
	)
	logger.info(
		f"Best MCS Association F1: {best_mcs_assoc_f1:.1%} "
		f"at Membership Threshold = {best_mcs_thresh:.1f}, "
		f"Validity Threshold = {best_mcs_thresh:.1f}"
	)
	# Save results to JSON
	with open(os.path.join(output_dir, "results.json"), "w") as f:
		json.dump(serialized_results, f, indent=4)

if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	)
	main()
