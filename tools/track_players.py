import argparse
import logging
from pathlib import Path

from spltrack.pipeline import TrackingPipeline
from spltrack.config import get_default_cfg
from spltrack.utils import ImageSequence
from spltrack.visualization import TrackRenderer


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir",
        type=lambda p: Path(p).resolve(strict=True),
        help="""
Path to the folder containing the data.
Expected folder structure:
    data_dir/
    ├─ images/
    ├─ gc/
    │  ├─ logs.g.json
    │  ├─ logs.tc.json
    """.strip(),
    )

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=lambda p: Path(p).resolve(strict=True),
        help="Path to the tracking pipeline config to use.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    # Validate data dir
    data_dir_path: Path = args.data_dir
    assert data_dir_path.joinpath("images").is_dir()
    assert data_dir_path.joinpath("gc").is_dir()

    cfg = get_default_cfg()
    cfg.merge_from_file(str(args.config))

    pipeline = TrackingPipeline(cfg)

    res = pipeline.run(data_dir_path)

    logging.info("Rendering tracking results.")
    image_sequence = ImageSequence(data_dir_path / "images")
    video_file_path = Path("video.avi")
    renderer = TrackRenderer(30, (1920, 1080))
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
        annotations={},
    )
