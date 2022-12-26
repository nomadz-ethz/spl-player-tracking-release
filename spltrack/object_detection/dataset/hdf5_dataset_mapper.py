import copy
from pathlib import Path
from typing import List, Union

import cv2 as cv
import h5py
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


class HDF5DatasetMapper:
    """
    Adapted from DatasetMapper in detectron2.data.dataset_mapper
    to load images from a hdf5 file.
    """

    def __init__(
        self,
        images_hdf5_file_path: Path,
        is_train: bool,
        image_format: str = "BGR",
        augmentations: List[Union[T.Augmentation, T.Transform]] = [],
    ):
        # Load images hdf5 file in memory
        self.images_hf = h5py.File(str(images_hdf5_file_path), "r")
        self.is_train = is_train
        self.image_format = image_format
        self.augmentations = T.AugmentationList(augmentations)

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        annos = [
            utils.transform_instance_annotations(
                obj,
                transforms,
                image_shape,
            )
            for obj in dataset_dict.pop("annotations")
        ]
        instances = utils.annotations_to_instances(
            annos,
            image_shape,
        )

        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = cv.imdecode(
            np.array(self.images_hf[dataset_dict["file_name"]]),
            cv.IMREAD_COLOR,
        )
        if self.image_format == "RGB":
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            # Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
