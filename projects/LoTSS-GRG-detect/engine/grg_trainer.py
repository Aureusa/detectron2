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

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.samplers import InferenceSampler

# Import custom modules from parent directory
sys.path.insert(0, str(project_root))
from data.dataset_mapper import GRGDatasetMapper as NPZProposalDatasetMapper

logger = logging.getLogger("LoTSS-GRG-detect.train")


class GRGTrainer(DefaultTrainer):
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

        # For now we use COCOEvaluator; will be replaced with custom evaluator
        return COCOEvaluator(
            dataset_name,
            output_dir=output_folder,
            tasks=("bbox", "segm")
        )
    
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
        mapper = NPZProposalDatasetMapper(
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
        
        mapper = NPZProposalDatasetMapper(
            cfg,
            is_train=False
        )
        return build_detection_test_loader(
            dataset=dataset_dicts,
            mapper=mapper,
            sampler=InferenceSampler(len(dataset_dicts)),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )
    