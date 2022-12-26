from pathlib import Path

import torch
from detectron2.utils import comm
from detectron2.solver import build_lr_scheduler
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import (
    hooks,
    DefaultTrainer,
)

from .hooks import (
    EvaluationVisualizationHook,
    ReduceLROnPlateauLRScheduler,
)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(
            dataset_name,
            output_dir=str(Path(cfg.OUTPUT_DIR).joinpath("eval")),
        )

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        if cfg.SOLVER.LR_SCHEDULER_NAME == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                patience=5,
                min_lr=1e-6,
                threshold=1,
            )
        else:
            return build_lr_scheduler(cfg, optimizer)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()

        ret = [
            hooks.IterationTimer(),
        ]

        if cfg.SOLVER.LR_SCHEDULER_NAME != "ReduceLROnPlateau":
            ret.append(hooks.LRScheduler())

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        # Add predictions visualization hook
        ret.append(EvaluationVisualizationHook(cfg.TEST.EVAL_VIS_PERIOD))

        # LR on plateau hook
        if cfg.SOLVER.LR_SCHEDULER_NAME == "ReduceLROnPlateau":

            def get_ap_ball():
                return self._last_eval_results["bbox"]["AP-ball"]

            ret.append(ReduceLROnPlateauLRScheduler(cfg.TEST.EVAL_PERIOD, get_ap_ball))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
