import torch
from torch.utils.data import Dataset, Subset


def subsample_dataset(dataset: Dataset, step: int):
    length = len(dataset)
    step = min(length, step)
    indices = torch.linspace(0, length - 1, step, dtype=int)
    return Subset(dataset, indices)
