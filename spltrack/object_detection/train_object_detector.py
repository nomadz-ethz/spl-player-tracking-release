import os
from datetime import datetime
from pathlib import Path

import wandb

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import verify_results


from .trainer import Trainer
from .dataset import register_orc_rc22_object_detection_dataset


def setup_cfg(args):
    # Get base Faster-RCNN config and config checkpoint
    cfg = get_cfg()

    # custom option for the orc dataset
    cfg.DATASETS.DOWNSAMPLING_STEP = 0
    cfg.TEST.EVAL_VIS_PERIOD = 1000
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.resume:
        run_output_dir_path = args.output_dir
    else:
        run_output_dir_path = args.output_dir / args.run_id
        run_output_dir_path.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(run_output_dir_path)

    return cfg


def main(args):

    cfg = setup_cfg(args)

    default_setup(cfg, args)

    # Register the dataset
    register_orc_rc22_object_detection_dataset(
        Path(os.getenv("DATASETS", "datasets")) / "orc_rc22",
        downsampling_step=cfg.DATASETS.DOWNSAMPLING_STEP,
    )

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    # Configure trainer
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


def _parse_args():
    parser = default_argument_parser(epilog="Train object detector using Detectron2")

    parser.add_argument(
        "--output-dir",
        required=True,
        type=lambda p: Path(p).absolute(),
        help="Path to the directory where artifacts should be saved.",
    )

    parser.add_argument(
        "--no-wandb", action="store_true", help="disable wandb logging."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.resume or args.eval_only:
        args.run_id = Path(args.output_dir).name
    else:
        args.run_id = (
            Path(args.config_file).stem
            + "_"
            + datetime.now().strftime("%d-%m-%Y_%H-%M")
        )

    if not args.no_wandb and args.num_gpus > 1:
        print("wandb logging disabled - not implemented for multi-gpu")
        args.no_wandb = True

    if not args.no_wandb:
        args.output_dir.mkdir(exist_ok=True, parents=True)
        wandb.init(
            project="orc_object_detector",
            dir=str(args.output_dir.absolute()),
            entity="nomadz_rc22",
            config=setup_cfg(args),
            settings=wandb.Settings(start_method="fork"),
        )
        tb_root_log_dir = args.output_dir.absolute() / "tensorboard"
        tb_root_log_dir.mkdir(parents=True, exist_ok=True)
        wandb.tensorboard.patch(
            root_logdir=str(tb_root_log_dir),
            pytorch=True,
        )

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
