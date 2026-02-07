#!/usr/bin/env python
"""
Training script for GRG detection using Mask R-CNN with precomputed proposals.

Usage:
    python train.py --config-file ../configs/mask_rcnn_R_50_FPN_grg.yaml
"""

import os
import sys
import logging
from pathlib import Path

# Add detectron2 to path (assumes detectron2 is in the parent directory structure)
detectron2_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(detectron2_root))

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

# Import custom modules from parent directory
sys.path.insert(0, str(project_root))
from data.register_dataset import main as register_datasets
from engine.grg_trainer import GRGTrainer

logger = logging.getLogger("LoTSS-GRG-detect.train")


def setup(args):
    """
    Create config and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Ensure output directory exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    default_setup(cfg, args)
    
    # Register datasets
    dataset_config_path = os.path.join(
        os.path.dirname(args.config_file), 
        "..", 
        "config", 
        "dataset.yaml"
    )
    dataset_config_path = os.path.normpath(dataset_config_path)
    
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
        MetadataCatalog.get(dataset_name).set(
            thing_classes=["GRG"],
            proposal_dir=proposal_dir
        )
        logger.info(f"Set proposal_dir for {dataset_name}: {proposal_dir}")
    
    return cfg


def main(args):
    """
    Main training function.
    """
    cfg = setup(args)
    
    # Log configuration
    logger.info("Training configuration:")
    logger.info(f"  Model: Mask R-CNN with ResNet-50 FPN")
    logger.info(f"  Training dataset: {cfg.DATASETS.TRAIN}")
    logger.info(f"  Validation dataset: {cfg.DATASETS.TEST}")
    logger.info(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    logger.info(f"  Base learning rate: {cfg.SOLVER.BASE_LR}")
    logger.info(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
    logger.info(f"  Evaluation period: {cfg.TEST.EVAL_PERIOD}")
    logger.info(f"  Output directory: {cfg.OUTPUT_DIR}")
    logger.info(f"  Precomputed proposals: {cfg.MODEL.LOAD_PROPOSALS}")
    logger.info(f"  Proposal top-k (train): {cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN}")
    logger.info(f"  Proposal top-k (test): {cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST}")
    
    if args.eval_only:
        model = GRGTrainer.build_model(cfg)
        GRGTrainer.test(cfg, model)
        return
    
    # Start training
    trainer = GRGTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting GRG detection training")
    logger.info(f"Command line args: {args}")
    
    # Launch training (supports distributed training)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
