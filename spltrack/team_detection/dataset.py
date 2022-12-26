import torch
import cv2
from torch.utils.data import Dataset
import pandas as pd


class RobotTeams(Dataset):
    def __init__(self, csv_file, transform=None):
        self.transform = transform
        self.images = pd.read_csv(csv_file)[::20]
        self.images = self.images[self.images.color >= 0]
        self.images.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename, label, x_min, y_min, x_max, y_max, color, number = self.images.iloc[
            idx
        ]
        image = cv2.imread(filename)
        if image is None:
            return None
        try:
            image = image[int(y_min) : int(y_max), int(x_min) : int(x_max)]
            if image is None:
                return None
        except:
            None
        if self.transform:
            image = self.transform(image)
        return image, int(color)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
