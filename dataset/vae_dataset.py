import torch
import lightning.pytorch as pl
import numpy as np
import os
from typing import Optional, Union


class VAEDataset(torch.utils.data.Dataset):

    def __init__(self, path: str, channels: int, img_res: tuple[int, int]) -> None:
        super().__init__()

        self.path = path
        self.channels = channels
        self.img_res = img_res

        self.mean: np.ndarray = np.array(
            [5.4008180e+04, 2.7393976e+02, 2.3674958e-03, 2.7760931e+02, 6.2221546e+00], dtype=np.float32)
        self.std: np.ndarray = np.array(
            [3.3518201e+03, 1.5841744e+01, 2.4138752e-03, 2.1723423e+01, 3.4207888e+00], dtype=np.float32)

        self.data = self._load_data()
        # self.data = self._normalize()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        item = self.data[idx]

        normalized_item = (
            item - self.mean[:, None, None]) / self.std[:, None, None]

        return normalized_item

    def _load_data(self):

        path = self.path
        c = self.channels
        w, h = self.img_res

        data = np.memmap(path, mode="r", dtype=np.float32).reshape(-1, c, w, h)

        return data

    def _normalize(self):
        return (self.data - self.mean[None, :, None, None]) / self.std[None, :, None, None]

    def denormalize_sample(self, sample):
        return sample * self.std[None, :, None, None] + self.mean[None, :, None, None]
