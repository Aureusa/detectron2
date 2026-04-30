#!/usr/bin/env python
"""
Testing script for Mask R-CNN with ComponentAssociationEvaluator.

This script mirrors the notebook-style evaluation flow:
- setup imports/configuration
- register datasets from dataset config
- load model weights/checkpoint
- run one inference pass and cache predictions in evaluator
- sweep score thresholds by calling evaluator.evaluate() repeatedly
- save all thresholded results to JSON
"""

import datetime
import json
import logging
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn

# Add detectron2 to path (assumes detectron2 is in the parent directory structure)
detectron2_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(detectron2_root))

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup
from detectron2.evaluation import DatasetEvaluator, inference_context
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds, setup_logger

from configs.custom_config import add_lotss_grg_config
from data.dataset_mapper import GRGDatasetMapper
from data.register_dataset import main as register_datasets
from evaluation.grg_evaluator import ComponentAssociationEvaluator

import pipelines.train

logger = setup_logger(name="LoTSS-GRG-detect.test_masked_rcnn_for_compassoc", termcolor="magenta")


def setup(args):
    """
    Create config and perform basic setup.
    """
    cfg = get_cfg()
    add_lotss_grg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.dataset_config:
        dataset_config_path = os.path.normpath(args.dataset_config)
    else:
        raise ValueError("Dataset config file must be provided with --dataset-config")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    default_setup(cfg, args)

    logger.info(f"Registering datasets from: {dataset_config_path}")
    registered = register_datasets(dataset_config_path)
    logger.info(f"Registered datasets: {registered}")

    # Keep this metadata setup since some datasets/mappers rely on it.
    for dataset_name in registered:
        split = dataset_name.split("_")[-1]  # train, val, or test
        proposal_dir = os.path.join(
            cfg.DATASETS.get("DATA_ROOT", "/home/s4861264/project_data/full-dataset"),
            split,
            "proposals",
        )
        MetadataCatalog.get(dataset_name).set(proposal_dir=proposal_dir)
        logger.info(f"Set proposal_dir for {dataset_name}: {proposal_dir}")

    return cfg


def process_with_evaluator(model, data_loader, evaluator: Union[DatasetEvaluator, ComponentAssociationEvaluator], callbacks=None):
    """
    Run model on data_loader once and cache predictions inside evaluator.
    """
    num_devices = get_world_size()
    local_logger = logging.getLogger(__name__)
    local_logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)
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

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    local_logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )

    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    local_logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    return evaluator


def _find_test_dataset_name(registered_datasets, dataset_type="val"):
    if not registered_datasets:
        raise ValueError("No datasets were registered from dataset config.")

    candidates = [d for d in registered_datasets if dataset_type in d.lower()]
    if len(candidates) >= 1:
        return candidates[0]

    return registered_datasets[-1]


def _serialize_results(results):
    """Convert results dict into JSON-serializable structure."""

    def _to_jsonable(value):
        if isinstance(value, dict):
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
    parser.add_argument(
        "--dataset-name-substring",
        default="lotss_grg",
        help="Substring to select target registered dataset names.",
    )
    parser.add_argument(
        "--score-threshold-step",
        type=float,
        default=0.1,
        help="Step size for score-threshold sweep in [0,1].",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = setup(args)

    output_dir = args.output_dir if args.output_dir else cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    config_file = os.path.abspath(args.config_file)
    dataset_config = os.path.abspath(args.dataset_config)

    all_registered_dataset = DatasetCatalog.list()
    registered_dataset = [d for d in all_registered_dataset if args.dataset_name_substring in d]
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

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Model config not found: {config_file}")
    if not os.path.exists(dataset_config):
        raise FileNotFoundError(f"Dataset config not found: {dataset_config}")

    training_info = "Testing configuration:"
    training_info += f"\n  -Model: Mask R-CNN with ResNet-50 FPN"
    training_info += f"\n  -Dataset used: {cfg.DATASETS.TEST}"
    training_info += f"\n  -Results directory: {output_dir}"
    training_info += f"\n  -Model config: {config_file}"
    training_info += f"\n  -Dataset config: {dataset_config}"
    training_info += f"\n  -Model weights: {cfg.MODEL.WEIGHTS}"
    training_info += f"\n  -Device: {cfg.MODEL.DEVICE}"
    logger.info(training_info)

    test_data = DatasetCatalog.get(test_dataset_name)
    logger.info(f"Number of test images: {len(test_data)}")
    if len(test_data) == 0:
        raise RuntimeError(f"Dataset {test_dataset_name} is empty.")

    predictor = DefaultPredictor(cfg)
    logger.info("Model loaded successfully")

    annotations_path = MetadataCatalog.get(test_dataset_name).json_file
    evaluator = ComponentAssociationEvaluator(
        coco_images=test_data,
        annotations_path=annotations_path,
        score_threshold=0.0,
    )

    test_data = DatasetCatalog.get(test_dataset_name)

    test_loader = build_detection_test_loader(
        dataset=test_data,
        mapper=GRGDatasetMapper(cfg, is_train=False),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

    logger.info(
        "Running inference on {} batches ({} images) with {} devices...".format(
            len(test_loader),
            len(test_data),
            torch.cuda.device_count(),
        )
    )

    evaluator = process_with_evaluator(
        predictor.model,
        test_loader,
        evaluator,
        callbacks=None,
    )

    step = max(1e-6, float(args.score_threshold_step))
    score_thresholds = np.arange(0.0, 1.0 + step, step)

    best_segm_source_f1 = -1.0
    best_segm_component_f1 = -1.0
    best_bbox_source_f1 = -1.0
    best_bbox_component_f1 = -1.0

    best_segm_source_thresh = None
    best_segm_component_thresh = None
    best_bbox_source_thresh = None
    best_bbox_component_thresh = None

    results = {}
    for score_thresh in score_thresholds:
        score_thresh = float(min(1.0, score_thresh))
        key = f"score_threshold_{score_thresh:.2f}"

        evaluator.set_score_threshold(score_thresh)
        result = evaluator.evaluate(flatten_results=False)
        results[key] = result

        curr = result.get("CAE", {})

        segm_source_f1 = float(curr.get("segm_f1", 0.0))
        segm_component_f1 = float(curr.get("segm_component_f1", 0.0))
        bbox_source_f1 = float(curr.get("bbox_f1", 0.0))
        bbox_component_f1 = float(curr.get("bbox_component_f1", 0.0))

        if segm_source_f1 > best_segm_source_f1:
            best_segm_source_f1 = segm_source_f1
            best_segm_source_thresh = score_thresh

        if segm_component_f1 > best_segm_component_f1:
            best_segm_component_f1 = segm_component_f1
            best_segm_component_thresh = score_thresh

        if bbox_source_f1 > best_bbox_source_f1:
            best_bbox_source_f1 = bbox_source_f1
            best_bbox_source_thresh = score_thresh

        if bbox_component_f1 > best_bbox_component_f1:
            best_bbox_component_f1 = bbox_component_f1
            best_bbox_component_thresh = score_thresh

    serialized_results = _serialize_results(results)

    logger.info(
        f"Best Segm Source F1: {best_segm_source_f1:.1%} at Score Threshold = {best_segm_source_thresh:.2f}"
    )
    logger.info(
        f"Best Segm Component F1: {best_segm_component_f1:.1%} at Score Threshold = {best_segm_component_thresh:.2f}"
    )
    logger.info(
        f"Best BBox Source F1: {best_bbox_source_f1:.1%} at Score Threshold = {best_bbox_source_thresh:.2f}"
    )
    logger.info(
        f"Best BBox Component F1: {best_bbox_component_f1:.1%} at Score Threshold = {best_bbox_component_thresh:.2f}"
    )

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(serialized_results, f, indent=4)

    logger.info(f"Saved threshold sweep results to {results_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
