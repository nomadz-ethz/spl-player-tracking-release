import argparse
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

from spltrack.config import get_default_cfg
from spltrack.tracking import Tracktor
from spltrack.utils import save_tracks


def main(
    images_dir_path: Path,
    config_file_path: Path,
    output_dir_path: Path,
    visualize: bool = False,
):
    images = sorted(images_dir_path.glob("*.jpg"))

    cfg = get_default_cfg()
    cfg.merge_from_file(str(config_file_path))

    tracktor = Tracktor.from_config(cfg)

    output_dir_path.mkdir(exist_ok=True, parents=True)

    if visualize:
        video_file_path = output_dir_path / "tracklets.avi"
        video_writer = cv.VideoWriter(
            str(video_file_path),
            cv.VideoWriter_fourcc(*"mp4v"),
            30,
            (1920, 1080),
        )

    for frame_id, image_file_path in enumerate(tqdm(images)):
        image = cv.imread(str(image_file_path))

        active_tracks_info = tracktor.track(frame_id, image)

        if visualize:
            for ti in active_tracks_info:
                x1, y1, x2, y2 = ti["bbox"].astype(int)
                track_id = ti["track_id"]
                color = (255, 0, 0) if track_id != -1 else (255, 255, 255)
                cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv.putText(
                    image,
                    f"{track_id}",
                    (int(x1), int(y1)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
            video_writer.write(image)

    if visualize:
        video_writer.release()

    tracklets = tracktor.get_all_tracks()
    tracks_file_path = output_dir_path / "tracklets.csv"
    save_tracks(tracks_file_path, tracklets)


def _parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "images_dir",
        type=lambda p: Path(p).expanduser().resolve(strict=True),
    )

    parser.add_argument(
        "-c",
        "--config",
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=lambda p: Path(p).expanduser(),
        required=True,
    )

    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Visualize results for every frame and save them to <output_dir>/viz",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        args.images_dir,
        args.config,
        args.output_dir,
        args.visualize,
    )
