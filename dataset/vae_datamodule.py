import torch
import lightning.pytorch as pl
import numpy as np
import os
from typing import Optional, Union

from dataset.vae_dataset import VAEDataset

class VAEDataModule(pl.LightningDataModule):
    def __init__(self,train_path: str, val_path: str, test_path: str, img_channel: int, img_res: tuple[int, int], batch_size: int, num_workers: int) -> None:
        super().__init__()

        self.train_path: str = train_path
        self.val_path: str = val_path
        self.test_path: str = test_path

        self.img_channel: int = img_channel
        self.img_res: tuple[int, int] = img_res
        
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        if self.num_workers <= 0:
            self.prefetch_factor = None
        else:
            self.prefetch_factor = 2

        self.train_ds: Optional[VAEDataset] = None
        self.val_ds: Optional[VAEDataset] = None
        self.test_ds: Optional[VAEDataset] = None

    def prepare_data(self) -> None:
        print("Data preparation complete.")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_ds = VAEDataset(
                path=self.train_path,
                channels=self.img_channel,
                img_res=self.img_res
            )

            self.val_ds = VAEDataset(
                path=self.val_path,
                channels=self.img_channel,
                img_res=self.img_res
            )
        if stage == "test" or stage is None:
            self.test_ds = VAEDataset(
                path=self.test_path,
                channels=self.img_channel,
                img_res=self.img_res
            )

        print("Data setup complete.")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )
