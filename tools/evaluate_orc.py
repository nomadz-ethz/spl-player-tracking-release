import argparse
import logging
from datetime import datetime
import shutil
from pathlib import Path
import wandb
import numpy as np
import json
from tqdm import tqdm

from spltrack.config import get_default_cfg
from spltrack.data.annotations import load_csv_annotations
from spltrack.data.utils import find_sequence_dirs
from spltrack.pipeline import TrackingPipeline
from spltrack.visualization import TrackRenderer
from spltrack.evaluation.metrics import (
    mean_player_identification_accuracy,
    team_detection_precision,
)
from spltrack.utils import ImageSequence
from spltrack.data.constants import ORC_SPLITS


def evaluate(
    cfg,
    dataset_dir,
    output_dir_path,
    task="train",
    visualize=False,
    quiet=False,
):
    logging.info("Initializing pipeline")
    pipeline = TrackingPipeline(cfg)

    sequence_dirs = find_sequence_dirs(dataset_dir, ORC_SPLITS[task])
    if len(sequence_dirs) == 0:
        logging.warning(f"No data directories found at {str(dataset_dir)}")
        return None
    else:
        # Optimize and compute mean player identification accuracy for every sequence
        tracking_metrics = {"mpia": {}, "tdp": {}}
        if visualize:
            renderer = TrackRenderer(30, (1920, 1080), footer_height=550)
        for d in tqdm(sequence_dirs, disable=quiet):
            logging.info(f"Processing sequence {d.name}")
            sequence_output_dir = output_dir_path / d.name
            sequence_output_dir.mkdir(exist_ok=True)
            res = pipeline.run(d, sequence_output_dir)
            try:
                annotations_file_path = next(d.glob("*.csv"))
            except StopIteration:
                logging.warning(
                    f"No annotations found for sequence {d.name}. Skipping evaluation."
                )
                continue
            annotations_per_image = load_csv_annotations(annotations_file_path)
            # Compute evaluation metrics
            seq_valid_name = d.name.strip("_")
            mpia = mean_player_identification_accuracy(
                tracking_results=res,
                annotations_per_image=annotations_per_image,
            )
            tracking_metrics["mpia"][seq_valid_name] = mpia

            tdp = team_detection_precision(
                tracking_results=res,
                annotations_per_image=annotations_per_image,
            )
            tracking_metrics["tdp"][seq_valid_name] = tdp

            if visualize:
                logging.info("Rendering tracking results.")
                image_sequence = ImageSequence(d / "images")
                video_file_path = sequence_output_dir / "video.avi"
                renderer.render(
                    video_file_path=video_file_path,
                    image_sequence=image_sequence,
                    tracks=res["tracks"],
                    excluded_tracks=res["excluded_tracklets"],
                    tracklets=res["tracklets"],
                    track_id_to_player=res["track_id_to_player"],
                    team_info=res["team_info"],
                    team_comm_poses_by_ts=res["team_comm_poses_by_ts"],
                    tracking_data=res["tracking_data"],
                    tracklet_to_fallen_events=res["tracklet_to_fallen_events"],
                    tracklet_to_track_cost_details=res[
                        "tracklet_to_track_cost_details"
                    ],
                    tracklet_to_track_id=res["tracklet_to_track_id"],
                    tracklet_to_tracklet_costs=res["tracklet_to_tracklet_costs"],
                    annotations=annotations_per_image,
                )
        tracking_metrics["mpia"]["average"] = np.mean(
            list(tracking_metrics["mpia"].values())
        )
        tracking_metrics["tdp"]["average"] = np.mean(
            list(tracking_metrics["tdp"].values())
        )
        with open(output_dir_path / "metrics.json", "w") as f:
            json.dump(tracking_metrics, f, indent=4)
        return tracking_metrics


def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset_dir",
        required=True,
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        help="Path to the dataset directory",
    )

    parser.add_argument(
        "-c",
        "--config-file",
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        required=True,
        help="Path to the config file",
    )

    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=list(ORC_SPLITS.keys()),
        default="train",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=lambda p: Path(p).expanduser(),
        help="Path to the output directory.",
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging.",
    )

    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Visualize optimized tracks for each sequence",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set log level to info",
    )

    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if not args.no_wandb:
        wandb.init(
            config=args,
            project="orc_tracking_pipeline",
            entity="nomadz_rc22",
            settings=wandb.Settings(start_method="fork"),
        )
        config = wandb.config
        run_name = wandb.run.name
        if run_name is None:
            run_name = datetime.strftime(datetime.now(), "%y-%m-%d_%H-%M")

    cfg = get_default_cfg()
    cfg.merge_from_file(str(args.config_file))

    if not args.no_wandb:
        cfg.merge_from_list(config.opts)
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir_path = args.output_dir / run_name
    output_dir_path.mkdir(exist_ok=True, parents=True)

    shutil.copy(args.config_file, output_dir_path / "config.yaml")

    logging.basicConfig(level=logging.INFO)
    metrics = evaluate(
        cfg,
        args.dataset_dir,
        output_dir_path,
        args.task,
        args.visualize,
    )
    if not args.no_wandb and metrics is not None:
        wandb.log(metrics)
