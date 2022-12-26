import torch

from detectron2.engine import HookBase, LRScheduler
from detectron2.utils import comm
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper
from .dataset.utils import subsample_dataset


class EvaluationVisualizationHook(HookBase):
    def __init__(self, period: int, *args):
        self._period = period

    def _run(self):
        cfg = self.trainer.cfg.clone()
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        data_items = subsample_dataset(
            DatasetCatalog.get(cfg.DATASETS.TEST[0]), step=100
        )
        data_mapper = DatasetMapper(cfg, is_train=False)

        model = self.trainer.model
        model.eval()

        for data_item in data_items:
            inputs = data_mapper(data_item)
            with torch.no_grad():
                predictions = model([inputs])[0]

            image_tensor = inputs["image"]
            image = image_tensor.cpu().numpy().transpose(1, 2, 0)
            if cfg.INPUT.FORMAT == "BGR":
                image = image[..., ::-1]
            visualizer = Visualizer(
                image, metadata=metadata, instance_mode=ColorMode.SEGMENTATION
            )
            instances = predictions["instances"].to("cpu")
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
            vis_image = vis_output.get_image().transpose(2, 0, 1)

            self.trainer.storage.put_image(data_item["file_name"], vis_image)

        model.train()

        comm.synchronize()

    def after_step(self):

        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._run()
        if self.trainer.iter % self._period != 0:
            return


class ReduceLROnPlateauLRScheduler(LRScheduler):
    def __init__(self, eval_period: int, get_metric_fn: callable):
        super().__init__()
        self._period = eval_period
        self._get_metric_fn = get_metric_fn

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        assert isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self._scheduler = self.scheduler
        self._best_param_group_id = LRScheduler.get_best_param_group_id(self._optimizer)

    def _run(self):
        metric = self._get_metric_fn()
        self._scheduler.step(metric)

    def after_step(self):
        # Log learning rate
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            self._run()
        if self.trainer.iter % self._period != 0:
            return
