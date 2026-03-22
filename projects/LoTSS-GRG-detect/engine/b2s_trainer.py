import os
import torch
import numpy as np
import time
import sys
import logging
from pathlib import Path

# Add detectron2 to path (assumes detectron2 is in the parent directory structure)
detectron2_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(detectron2_root))

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detectron2.engine import DefaultTrainer
from detectron2.engine.train_loop import SimpleTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.samplers import InferenceSampler
import detectron2.utils.comm as comm
from detectron2.utils.events import get_event_storage

# Import custom modules from parent directory
from data.dataset_mapper import GRGDatasetMapper as NPZProposalDatasetMapper
from data.dataset_mapper import B2SDatasetMapper as B2SNPZProposalDatasetMapper
from evaluation.b2s_evaluator import B2SEvaluator, B2SMultiClassEvaluator
from engine.backbone_freeze_hook import BackboneFreezeHook

logger = logging.getLogger("LoTSS-GRG-detect.train")


class B2STrainer(DefaultTrainer):
    """
    Custom trainer that uses NPZ proposals and evaluates during training.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator using filtered annotations (no empty segmentations).
        """    
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        
        if cfg.TEST.EVALUATOR == "B2SEvaluator":
            evaluator = B2SEvaluator(
                validity_threshold=cfg.MODEL.VALIDITY_HEAD.SCORE_THRESH_TEST,
                membership_threshold=cfg.MODEL.MEMBERSHIP_HEAD.SCORE_THRESH_TEST
            )
        elif cfg.TEST.EVALUATOR == "B2SMultiClassEvaluator":
            evaluator = B2SMultiClassEvaluator(
                membership_threshold=cfg.MODEL.MEMBERSHIP_HEAD.SCORE_THRESH_TEST,
            )
        else:
            raise ValueError(
                f"Unsupported evaluator type: {cfg.TEST.EVALUATOR}."
                f" Must be 'B2SEvaluator' or 'B2SMultiClassEvaluator'. Check your config file.")

        return DatasetEvaluators([
            evaluator
        ])
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build training dataloader with custom NPZ proposal mapper.
        Filters out images with annotations that have empty segmentations.
        """
        # Get proposal directory from metadata
        dataset_name = cfg.DATASETS.TRAIN[0]
        
        logger.info(f"Building training dataloader. Dataset: {dataset_name}")
        
        # Load dataset dicts
        dataset_dicts = DatasetCatalog.get(dataset_name)

        # Create custom mapper
        mapper = B2SNPZProposalDatasetMapper(
            cfg, 
            is_train=True,
        )
        return build_detection_train_loader(cfg, dataset=dataset_dicts, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Build test/validation dataloader with custom NPZ proposal mapper.
        Uses the same filtered dataset as the evaluator.
        """
        logger.info(f"Building test dataloader. Dataset: {dataset_name}")
        
        # Load dataset dicts
        dataset_dicts = DatasetCatalog.get(dataset_name)
        
        mapper = B2SNPZProposalDatasetMapper(
            cfg,
            is_train=False
        )
        return build_detection_test_loader(
            dataset=dataset_dicts,
            mapper=mapper,
            sampler=InferenceSampler(len(dataset_dicts)),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )

    def build_hooks(self):
        hooks = super().build_hooks()
        freeze_until_iter = int(self.cfg.SOLVER.BACKBONE_FREEZE_ITERS)
        if freeze_until_iter > 0:
            hooks.insert(0, BackboneFreezeHook(freeze_until_iter=freeze_until_iter))
        return hooks
