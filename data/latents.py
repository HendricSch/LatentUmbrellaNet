import dask
import xarray as xr
import numpy as np
import xbatcher as xb
import torch
from torchvision.transforms import Normalize
import lightning as pl
from tqdm import tqdm


def gen_latents_bgen(zarr_path: str, train: bool) -> xb.BatchGenerator:
    ds = xr.open_zarr(zarr_path, consolidated=False)

    num_samples = ds.sizes["time"]

    train_size = int(num_samples * 0.9)

    if train:
        ds = ds.isel(time=slice(0, train_size))

    else:
        ds = ds.isel(time=slice(train_size, num_samples))

    bgen = xb.BatchGenerator(
        ds,
        input_dims={"time": 3, "channel": 128, "x": 180, "y": 90},
        input_overlap={"time": 2, "channel": 0, "x": 0, "y": 0},
    )

    return bgen


class LatentsDataset(torch.utils.data.Dataset):
    """Dataset, das (x1, x2, y) Triple aus einem 3-Zeitpunkte-Fenster liefert."""

    def __init__(self, bgen: xb.BatchGenerator):
        super().__init__()

        self.bgen = bgen

    def __len__(self):
        return len(self.bgen)

    def __getitem__(self, idx):
        batch = self.bgen[idx]

        tensor = torch.tensor(batch.data.values)

        return tensor[0], tensor[1], tensor[2]


class LatentsDataModule(pl.LightningDataModule):
    """Lightning DataModule für latente Darstellungen."""

    def __init__(self, batch_size: int = 8):
        super().__init__()

        self.batch_size = batch_size

        bgen_train = gen_latents_bgen(
            zarr_path="zarr_files/latent_moments_vae-kl-f8.zarr", train=True
        )

        bgen_val = gen_latents_bgen(
            zarr_path="zarr_files/latent_moments_vae-kl-f8.zarr", train=False
        )

        self.train_ds = LatentsDataset(bgen=bgen_train)
        self.val_ds = LatentsDataset(bgen=bgen_val)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )


def main():
    data = LatentsDataModule(batch_size=8)

    dataloader = data.train_dataloader()

    for i, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)

        if i == 10:
            break


if __name__ == "__main__":
    main()
