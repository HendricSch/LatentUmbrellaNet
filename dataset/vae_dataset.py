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

        self.mean: np.ndarray = np.array([5.40695777e+04, 2.74162604e+02, 2.41138667e-03, 2.77659178e+02, 6.11351155e+00])
        self.std: np.ndarray = np.array([3.28407787e+03, 1.60311676e+01, 2.39652644e-03, 2.20706169e+01, 3.32614560e+00])

        self.data = self._load_data()
        self.data = self._normalize()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

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