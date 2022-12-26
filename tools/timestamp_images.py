import argparse
import shutil
from pathlib import Path

from spltrack.data.annotations import load_csv_annotations, save_csv_annotations

FPS = 29.97
frame_interval = 1000 * (1 / FPS)


def frame_id_to_timestamp(frame_id: int) -> float:
    return int((frame_id - 1) * frame_interval)


parser = argparse.ArgumentParser("Timestamp images and annotations.")
parser.add_argument(
    "--images",
    type=lambda p: Path(p).resolve(),
    default=None,
    help="Path to images directory",
)
parser.add_argument(
    "--annotations",
    type=lambda p: Path(p).resolve(),
    default=None,
    help="Path to annotations file",
)
args = parser.parse_args()

if args.images is not None:
    images_dir_path = Path(args.images)
    assert images_dir_path.is_dir()

    images = images_dir_path.glob("*.jpg")

    for image_file_path in images:
        frame_id = int(image_file_path.stem)
        timestamp = frame_id_to_timestamp(int(image_file_path.stem))
        new_image_file_path = image_file_path.with_stem(str(timestamp))
        shutil.move(image_file_path, new_image_file_path)

if args.annotations is not None:
    annotations_per_image = load_csv_annotations(args.annotations)
    timstamped_annotations_per_image = {}
    for filename, image_annotations in annotations_per_image.items():
        filename_p = Path(filename)
        frame_id = int(filename_p.stem)
        timestamp = frame_id_to_timestamp(frame_id)
        new_filename = str(timestamp) + filename_p.suffix
        timstamped_annotations_per_image[new_filename] = image_annotations

    save_csv_annotations(args.annotations, timstamped_annotations_per_image)
