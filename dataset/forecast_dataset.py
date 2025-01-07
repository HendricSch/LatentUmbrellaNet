import torch
import lightning.pytorch as pl
import numpy as np
import os
from typing import Optional, Union

from models.vqvae import VQVAE

class ForecastDataset(torch.utils.data.Dataset):

    def __init__(self, path: str, vqvae: VQVAE, channels: int, img_res: tuple[int, int]) -> None:
        super(ForecastDataset, self).__init__()

        self.vqvae = vqvae
        self.device = vqvae.device
        self.path = path

        self.channels = channels
        self.img_res = img_res

        self.data = self._load_data()

    def _load_data(self):
        path = self.path
        c = self.channels
        w, h = self.img_res

        data = np.memmap(path, mode="r", dtype=np.float32).reshape(-1, c, w, h)

        return data

    def __len__(self):
        return self.data.shape[0] - 1

    @torch.no_grad()
    def __getitem__(self, idx):
        x1 = self.data[idx]
        x2 = self.data[idx + 1]

        x1 = torch.tensor(x1, dtype=torch.float32).to(self.device)
        x2 = torch.tensor(x2, dtype=torch.float32).to(self.device)

        _, latent_x1, _ = self.vqvae(x1)
        _, latent_x2, _ = self.vqvae(x2)

        return latent_x1, latent_x2
    
if __name__ == "__main__":

    path = "train.memmap"
    channels = 5
    img_res = (128, 64)

    vqvae = VQVAE(in_channels=channels)

    dataset = ForecastDataset(path, vqvae, channels, img_res)
    

