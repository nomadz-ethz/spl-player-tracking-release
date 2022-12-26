import argparse
from pathlib import Path
from typing import Optional

from spltrack.utils import ImageSequence, load_tracks
from spltrack.visualization import TrackRenderer


def _main(
    tracks_file_path: Path,
    excluded_tracks_file_path: Optional[Path],
    images_dir_path: Path,
    output_dir_path: Optional[Path],
):
    if output_dir_path is None:
        output_dir_path = tracks_file_path.parent
    output_dir_path.mkdir(exist_ok=True)

    video_file_path = output_dir_path.joinpath(tracks_file_path.stem).with_suffix(
        ".avi"
    )

    # Load all the images
    image_sequence = ImageSequence(images_dir_path)

    tracks = load_tracks(tracks_file_path)
    if excluded_tracks_file_path is not None:
        excluded_tracks = load_tracks(excluded_tracks_file_path)
    else:
        excluded_tracks = None

    renderer = TrackRenderer(30, (1920, 1080))
    renderer.render(video_file_path, image_sequence, tracks, excluded_tracks)


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Read tracks from a file and render to video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "tracks_file",
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        help="Path to tracklets file",
    )

    parser.add_argument(
        "images_dir",
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        help="Path to the directory containing the raw images.",
    )

    parser.add_argument(
        "--excluded-tracks",
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        help="Path to excluded tracks file",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=lambda p: Path(p).expanduser(),
        help="Path to the output directory where the video should be saved.",
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _main(
        args.tracks_file,
        args.excluded_tracks,
        args.images_dir,
        args.output_dir,
    )
