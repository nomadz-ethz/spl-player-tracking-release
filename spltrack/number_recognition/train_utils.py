import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Subset, WeightedRandomSampler


def get_class_balanced_subset(dataset, n_samples_per_class: int = 3):

    labels = dataset.labels
    classes = np.unique(labels)
    # Get the indices of the first n samples for each class
    indices = []
    for c in classes:
        c_samples_indices = np.nonzero(labels == c)[0]
        c_selected_indices = []
        for idx in c_samples_indices:
            if len(c_selected_indices) == n_samples_per_class:
                break
            if len(c_selected_indices) == 0:
                c_selected_indices.append(idx)
            elif (idx - c_selected_indices[-1]) > 1000:
                c_selected_indices.append(idx)
        indices.append(c_selected_indices)
    # Flatten column major
    indices = np.hstack(indices).flatten("F")
    return Subset(dataset, indices=indices)


def get_inverse_preprocessing_transforms():
    return [
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
        ),
        T.ToPILImage(),
    ]


def get_preprocessing_transforms():
    return [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


def get_weighted_random_sampler(dataset):

    labels = dataset.labels

    class_sample_count = np.array(
        [len(np.where(labels == t)[0]) for t in np.unique(labels)]
    )

    weight_not_a_number = 1.0 / class_sample_count[0]
    weight_is_a_number = 1.0 / sum(class_sample_count[1:])

    samples_weights = np.array(
        [weight_is_a_number if l != 0 else weight_not_a_number for l in labels]
    )

    samples_weights = torch.from_numpy(samples_weights).double()

    return WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True,
    )
