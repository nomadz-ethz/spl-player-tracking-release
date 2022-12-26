import csv
import functools
from collections import defaultdict
from pathlib import Path
from typing import List

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances

# Image parameters
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

CSV_ANNOTATION_FIELDS = [
    "filename",
    "label",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "color",
    "number",
]

OBJECT_DETECTION_SPLITS = {
    "train": [
        "nao_devils",
        "robo_eireann",
        "spqr",
        "nomadz",
        "r2k",
    ],
    "val": [
        "berlin_united",
        "bhuman",
    ],
}


def load_object_detection_annotations_from_csv(annotations_file_path: Path):
    """Loads per image annoations from a CSV file

    :returns: a dict mapping image file names to a list of annotations
    """

    annotations_per_image = defaultdict(list)
    with annotations_file_path.open("r") as f:
        annotations_reader = csv.DictReader(f, fieldnames=CSV_ANNOTATION_FIELDS)
        for row in annotations_reader:
            filename = row["filename"]
            annotations_per_image[filename].append(
                {
                    "label": row["label"],
                    "x_min": row["x_min"],
                    "y_min": row["y_min"],
                    "x_max": row["x_max"],
                    "y_max": row["y_max"],
                }
            )
    return annotations_per_image


def get_orc_rc22_object_detection_data_dicts(
    data_root_dir_path: Path,
    sequences: List[str],
    downsampling_step: int = 0,
):

    items = []
    next_item_index = 0
    # Iterate over all the subdirectories - sort for deterministic behavior
    for data_dir_path in sorted(data_root_dir_path.iterdir()):
        # Sanity check if it's a valid data dir
        if data_dir_path.name not in sequences:
            continue

        # Load annotations
        annotations_file_path = next(data_dir_path.glob("*.csv"))
        annotations_per_image = load_object_detection_annotations_from_csv(
            annotations_file_path
        )

        if downsampling_step > 0:
            sampled_images = sorted(annotations_per_image.keys())[::downsampling_step]
            annotations_per_image = {
                k: annotations_per_image[k] for k in sampled_images
            }

        for file_name, annotations in annotations_per_image.items():
            if not file_name.endswith(".jpg"):
                continue

            box_annotations = []
            for annotation in annotations:
                label = annotation["label"]
                x_min = float(annotation["x_min"])
                y_min = float(annotation["y_min"])
                x_max = float(annotation["x_max"])
                y_max = float(annotation["y_max"])
                box_annotations.append(
                    {
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": label,
                        "bbox": [x_min, y_min, x_max, y_max],
                    }
                )
            items.append(
                {
                    "file_name": str(data_dir_path.joinpath(f"images/{file_name}")),
                    "image_id": next_item_index,
                    "height": IMAGE_HEIGHT,
                    "width": IMAGE_WIDTH,
                    "annotations": box_annotations,
                }
            )
            next_item_index += 1

    return items


def register_orc_rc22_object_detection_dataset(
    data_root_dir_path: Path,
    downsampling_step: int = 0,
    use_coco_annots: bool = False,
):

    images_hdf5_file_path = data_root_dir_path.joinpath("images.h5")
    basename = "orc_rc22_object_detection"
    meta = {
        "images_hdf5_file": str(images_hdf5_file_path),
        "thing_classes": ["robot", "ball"],
        "thing_colors": [(0, 0, 255), (255, 0, 0)],
    }

    # Register the train and validation datasets
    for split, sequences in OBJECT_DETECTION_SPLITS.items():

        name = basename + "_" + split
        coco_instances_file = data_root_dir_path / f"annotations_coco_{split}.json"
        if use_coco_annots and coco_instances_file.is_file():
            register_coco_instances(
                name,
                meta,
                str(coco_instances_file),
                str(data_root_dir_path),
            )
        else:
            get_data_dicts_func = functools.partial(
                get_orc_rc22_object_detection_data_dicts,
                data_root_dir_path=data_root_dir_path,
                sequences=sequences,
                downsampling_step=downsampling_step,
            )
            DatasetCatalog.register(
                name=name,
                func=get_data_dicts_func,
            )
            MetadataCatalog.get(name).set(**meta)
