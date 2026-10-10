import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .config import DATA_PATH

MEAN, STD = 0.2860, 0.3530


def load_arrays():
    if not DATA_PATH.exists():
        sys.exit("[!] Dataset belum ada. Jalankan: python -m src.prepare_data && dvc add data/fmnist_subset.npz")
    return dict(np.load(DATA_PATH))


def normalize(x_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(x_uint8).float() / 255.0
    return ((x - MEAN) / STD).unsqueeze(1)  # (N,1,28,28)


class ImgDataset(Dataset):
    def __init__(self, x, y, augment=False):
        self.x = normalize(x)
        self.y = torch.from_numpy(y).long()
        self.tf = (v2.Compose([v2.RandomCrop(28, padding=2), v2.RandomHorizontalFlip()])
                   if augment else None)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.tf is not None:
            x = self.tf(x)
        return x, self.y[i]


def make_loaders(arrays, batch_size, augment, seed):
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(ImgDataset(arrays["x_train"], arrays["y_train"], augment),
                       batch_size=batch_size, shuffle=True, generator=g)
    val = DataLoader(ImgDataset(arrays["x_val"], arrays["y_val"]), batch_size=512)
    test = DataLoader(ImgDataset(arrays["x_test"], arrays["y_test"]), batch_size=512)
    return train, val, test
