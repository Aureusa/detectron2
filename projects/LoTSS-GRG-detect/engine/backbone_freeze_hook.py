import logging

from detectron2.engine.train_loop import HookBase

logger = logging.getLogger("LoTSS-GRG-detect.train.BackboneFreezeHook")


class BackboneFreezeHook(HookBase):
    """
    Freeze model.backbone parameters at training start and unfreeze after a target iteration.
    """

    def __init__(self, freeze_until_iter: int):
        self.freeze_until_iter = int(freeze_until_iter)
        self._is_frozen = False
        self._finished = False

    def _unwrap_model(self):
        model = self.trainer.model
        return model.module if hasattr(model, "module") else model

    def _set_backbone_trainable(self, trainable: bool) -> int:
        model = self._unwrap_model()
        if not hasattr(model, "backbone"):
            return -1

        num_params = 0
        for param in model.backbone.parameters():
            param.requires_grad = trainable
            num_params += param.numel()
        return num_params

    def before_train(self):
        if self.freeze_until_iter <= 0:
            self._finished = True
            return

        if self.trainer.iter < self.freeze_until_iter:
            num_params = self._set_backbone_trainable(False)
            if num_params < 0:
                logger.warning("Model has no backbone attribute; skipping hook.")
                self._finished = True
                return
            self._is_frozen = True
            logger.info(
                "Froze backbone (%d parameters) at iter=%d, will unfreeze at iter=%d.",
                num_params,
                self.trainer.iter,
                self.freeze_until_iter,
            )
            return

        # Resume or start after freeze threshold: ensure backbone is trainable.
        num_params = self._set_backbone_trainable(True)
        if num_params < 0:
            logger.warning("Model has no backbone attribute; skipping hook.")
        else:
            logger.info(
                "Iter=%d >= %d, backbone left trainable (%d parameters).",
                self.trainer.iter,
                self.freeze_until_iter,
                num_params,
            )
        self._finished = True

    def before_step(self):
        if self._finished or not self._is_frozen:
            return

        if self.trainer.iter >= self.freeze_until_iter:
            num_params = self._set_backbone_trainable(True)
            if num_params < 0:
                logger.warning("Model has no backbone attribute; cannot unfreeze.")
            else:
                logger.info(
                    "Unfroze backbone (%d parameters) at iter=%d.",
                    num_params,
                    self.trainer.iter,
                )
            self._is_frozen = False
            self._finished = True
