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
from evaluation.b2s_evaluator import B2SEvaluator

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

        return DatasetEvaluators([
            B2SEvaluator(
                validity_threshold=cfg.MODEL.VALIDITY_HEAD.SCORE_THRESH_TEST,
                membership_threshold=cfg.MODEL.MEMBERSHIP_HEAD.SCORE_THRESH_TEST
            )
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
    
    # TODO: Maybe fix this at some point?
    # The default trainer's training loop is pretty rigid and doesn't allow for much customization
    # without overriding the entire loop. For now, we can just rely on the default
    # loop and make sure our custom dataloaders and evaluators are used.
    # def run_step(self):
    #     self._trainer.iter = self.iter
    #     self._run_step(self._trainer)
    
    # def _run_step(self, trainer: SimpleTrainer):
    #     """
    #     Implement the standard training logic described above.
    #     """
    #     assert trainer.model.training, "[SimpleTrainer] model was changed to eval mode!"
    #     start = time.perf_counter()
    #     """
    #     If you want to do something with the data, you can wrap the dataloader.
    #     """
    #     data = next(trainer._data_loader_iter)
    #     data_time = time.perf_counter() - start

    #     if trainer.zero_grad_before_forward:
    #         """
    #         If you need to accumulate gradients or do something similar, you can
    #         wrap the optimizer with your custom `zero_grad()` method.
    #         """
    #         trainer.optimizer.zero_grad()

    #     """
    #     If you want to do something with the losses, you can wrap the model.
    #     """
    #     loss_dict = trainer.model(data)
    #     loss_membership = loss_dict.get("loss_membership", 0.0)
    #     loss_proposal_validity = loss_dict.get("loss_proposal_validity", 0.0)
    #     losses = loss_membership + loss_proposal_validity
        
    #     if not trainer.zero_grad_before_forward:
    #         """
    #         If you need to accumulate gradients or do something similar, you can
    #         wrap the optimizer with your custom `zero_grad()` method.
    #         """
    #         trainer.optimizer.zero_grad()
    #     losses.backward()

    #     trainer.after_backward()

    #     if trainer.async_write_metrics:
    #         # write metrics asynchronically
    #         trainer.concurrent_executor.submit(
    #             self._write_metrics, trainer, loss_dict, data_time, iter=trainer.iter
    #         )
    #     else:
    #         self._write_metrics(trainer, loss_dict, data_time)

    #     """
    #     If you need gradient clipping/scaling or other processing, you can
    #     wrap the optimizer with your custom `step()` method. But it is
    #     suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
    #     """
    #     trainer.optimizer.step()

    # def _write_metrics(
    #     self,
    #     trainer: SimpleTrainer,
    #     loss_dict,
    #     data_time: float,
    #     prefix: str = "",
    #     iter = None,
    # ) -> None:
    #     logger = logging.getLogger(__name__)

    #     iter = trainer.iter if iter is None else iter
    #     if (iter + 1) % trainer.gather_metric_period == 0:
    #         try:
    #             B2STrainer.write_metrics(loss_dict, data_time, iter, prefix)
    #         except Exception:
    #             logger.exception("Exception in writing metrics: ")
    #             raise

    # @staticmethod
    # def write_metrics(
    #     loss_dict,
    #     data_time: float,
    #     cur_iter: int,
    #     prefix: str = "",
    # ) -> None:
    #     """
    #     Args:
    #         loss_dict (dict): dict of scalar losses
    #         data_time (float): time taken by the dataloader iteration
    #         prefix (str): prefix for logging keys
    #     """
    #     metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
    #     metrics_dict["data_time"] = data_time

    #     storage = get_event_storage()
    #     # Keep track of data time per rank
    #     storage.put_scalar("rank_data_time", data_time, cur_iter=cur_iter)

    #     # Gather metrics among all workers for logging
    #     # This assumes we do DDP-style training, which is currently the only
    #     # supported method in detectron2.
    #     all_metrics_dict = comm.gather(metrics_dict)

    #     if comm.is_main_process():
    #         # data_time among workers can have high variance. The actual latency
    #         # caused by data_time is the maximum among workers.
    #         data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
    #         storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

    #         # average the rest metrics
    #         metrics_dict = {
    #             k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
    #         }
    #         loss_membership = loss_dict.get("loss_membership", 0.0)
    #         loss_proposal_validity = loss_dict.get("loss_proposal_validity", 0.0)
    #         total_losses_reduced = (loss_membership + loss_proposal_validity).cpu().detach().numpy()
    #         if not np.isfinite(total_losses_reduced):
    #             raise FloatingPointError(
    #                 f"Loss became infinite or NaN at iteration={cur_iter}!\n"
    #                 f"loss_dict = {metrics_dict}"
    #             )

    #         storage.put_scalar(
    #             "{}total_loss".format(prefix), total_losses_reduced, cur_iter=cur_iter
    #         )
    #         if len(metrics_dict) > 1:
    #             storage.put_scalars(cur_iter=cur_iter, **metrics_dict)
    