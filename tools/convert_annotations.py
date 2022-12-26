import argparse
from pathlib import Path
from spltrack.data.annotations import (
    load_csv_annotations,
    save_csv_annotations,
    load_cvat_video_annotations,
    save_cvat_video_annotations,
)


def _parse_args():

    parser = argparse.ArgumentParser(
        """
Convert annotations from source to target format. \\
Supported formats: CSV, CVAT Video.
        """
    )

    parser.add_argument(
        "annotations_file_path",
        type=lambda p: Path(p).resolve(strict=True),
        help="Path to annotations file.",
    )

    parser.add_argument(
        "images_dir_path",
        type=lambda p: Path(p).resolve(strict=True),
        help="Path to images directory.",
    )

    parser.add_argument(
        "--to",
        type=str,
        choices=["csv", "cvat_video"],
        required=True,
        help="Target format.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    source_format = args.annotations_file_path.suffix[1:]
    if source_format not in ["csv", "cvat_video"]:
        raise ValueError(f"Unsupported target format: {source_format}")

    if source_format == args.to:
        raise ValueError(f"Source and target formats are the same: {source_format}")

    if source_format == "csv":
        annotations_per_image = load_csv_annotations(args.annotations_file_path)
    elif source_format == "cvat_video":
        annotations_per_image = load_cvat_video_annotations(
            args.annotations_file_path,
            args.images_dir_path,
        )

    if args.to == "csv":
        save_csv_annotations(
            args.annotations_file_path.with_suffix(".csv"),
            annotations_per_image,
        )
    elif args.to == "cvat_video":
        save_cvat_video_annotations(
            args.annotations_file_path.with_suffix(".xml"),
            annotations_per_image,
            args.images_dir_path,
        )
