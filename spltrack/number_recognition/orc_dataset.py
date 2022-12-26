from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
import cv2 as cv
import h5py
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from ..data.annotations import load_csv_annotations

_ORC_SPLITS = {"train": ["bhuman", "nomadz"], "val": ["nao_devils"]}
_ANNOTATIONS_FILE_NAME = "annotations_with_number_visibility.csv"


class ORCNumberRecognition(torch.utils.data.Dataset):
    """
    Class for the number recognition dataset generated from the ORC
    video sequences. Samples are generated dynamically by cropping
    ground truth bounding boxes from the original images. Labels are
    integers between 0 and 5, where 0 represents no number and [1,5]
    represent the corresponding number visible in the box.
    """

    def __init__(
        self,
        root_dir_path: Path,
        split: str,
        transforms: Optional[List] = None,
        load_images_from_hdf5: bool = False,
        size: Union[int, Tuple[int, int]] = (80, 64),
        max_aspect_ratio: float = 1.0,
    ):
        self.root_dir_path = root_dir_path
        if split not in _ORC_SPLITS:
            raise ValueError(
                f"Invalid split: {split}. "
                f"Valid splits: {', '.join(_ORC_SPLITS.keys())}"
            )
        self.split = split
        self.load_images_from_hdf5 = load_images_from_hdf5
        self.size = size
        self.max_aspect_ratio = max_aspect_ratio

        if transforms is not None and len(transforms) > 0:
            self._transform = T.Compose(transforms)
        else:
            self._transform = None

        self._sequences = _ORC_SPLITS[self.split]
        self._annotations = ORCNumberRecognition.load_annotations(
            self.root_dir_path,
            self._sequences,
            self.max_aspect_ratio,
        )

        if load_images_from_hdf5 and self.root_dir_path.joinpath("images.h5").is_file():
            self._images_hf = h5py.File(self.root_dir_path.joinpath("images.h5"))

            def load_image_from_hdf5(sequence: str, filename: str) -> torch.Tensor:
                image_bgr = cv.imdecode(
                    self._images_hf[f"{sequence}/{filename}"][()], cv.IMREAD_UNCHANGED
                )
                return Image.fromarray(cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB))

            self._load_image_fn = load_image_from_hdf5
        else:

            def _load_image_from_file(sequence: str, filename: str) -> torch.Tensor:
                image_file_path = self.root_dir_path / sequence / "images" / filename
                return Image.open(image_file_path)

            self._load_image_fn = _load_image_from_file

    def __getitem__(self, idx):
        annot = self._annotations[idx]
        sequence = annot["sequence"]
        filename = annot["filename"]
        image = self._load_image_fn(sequence, filename)

        left, top = annot["x_min"], annot["y_min"]
        width, height = annot["x_max"] - left, annot["y_max"] - top

        # Crop and resize box
        cropped_image = TF.resized_crop(
            image,
            top,
            left,
            height,
            width,
            size=self.size,
        )

        if self._transform:
            cropped_image = self._transform(cropped_image)

        # Get label
        if annot["number_is_visible"]:
            label = annot["number"]
        else:
            label = 0
        return cropped_image, label

    def __len__(self):
        return len(self._annotations)

    @property
    def labels(self):
        return np.array(
            [a["number"] if a["number_is_visible"] else 0 for a in self._annotations]
        )

    @classmethod
    def load_annotations(
        cls,
        root_dir_path: Path,
        sequences: List[str],
        max_aspect_ratio: float = 1.0,
    ):
        annots = []
        for sname in sequences:
            sequence_data_dir_path = root_dir_path / sname
            annotations_file_path = sequence_data_dir_path / _ANNOTATIONS_FILE_NAME
            if not annotations_file_path.is_file():
                print(f"Warning: annotations not found for sequence {sname}")
                continue
            # Load robot annotations filtering out those with aspect ratio
            # more than 1 (i.e. horizontal boxes)
            sequence_annots = [
                item
                for sublist in load_csv_annotations(annotations_file_path).values()
                for item in sublist
            ]
            for annot in sequence_annots:
                # Ignore balls
                if annot["label"] != 0:
                    continue
                width = annot["x_max"] - annot["x_min"]
                height = annot["y_max"] - annot["y_min"]
                if (width / height) > max_aspect_ratio:
                    continue
                annot["sequence"] = sname
                sequence_annots.append(annot)
            annots += sequence_annots

        return annots
