 """DataModule für ERA5 aus dem ARCO Zarr-Store.

 Stellt einen `xbatcher.BatchGenerator` sowie PyTorch Lightning `DataModule`
 bereit.
 """

import dask
import xarray as xr
import numpy as np
import xbatcher as xb
import torch
from torchvision.transforms import Normalize
import lightning as pl


def gen_bgen(train: bool) -> xb.BatchGenerator:
    """Erzeugt einen BatchGenerator über den ARCO-ERA5 Datensatz.

    train=True liefert 1979-2022, andernfalls 2023-2024.
    """

    TIME_START_TRAIN = "1979-01-01"
    TIME_STOP_TRAIN = "2022-12-31"
    TIME_START_VAL = "2023-01-01"
    TIME_STOP_VAL = "2024-12-31"

    LEVEL = [50, 100, 150, 200, 250, 300,
             400, 500, 600, 700, 850, 925, 1000]
    VARS_SURFACE = ["2m_temperature", "10m_u_component_of_wind",
                    "10m_v_component_of_wind", "mean_sea_level_pressure"]
    VARS_ATMOSPHERE = ["temperature", "u_component_of_wind",
                       "v_component_of_wind", "geopotential", "specific_humidity"]

    ds = xr.open_zarr('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
                      chunks=None, storage_options=dict(token='anon'),)

    if train:
        ds = ds.sel(time=slice(TIME_START_TRAIN,
                               TIME_STOP_TRAIN))  # 1979-2022
    else:
        ds = ds.sel(time=slice(TIME_START_VAL, TIME_STOP_VAL))  # 2023-2024

    ds = ds.sel(level=LEVEL)
    ds = ds[VARS_SURFACE + VARS_ATMOSPHERE]

    bgen = xb.BatchGenerator(
        ds, input_dims={"time": 1, "level": 13, "latitude": 721, "longitude": 1440})

    return bgen


class ERA5Dataset(torch.utils.data.Dataset):
    """Dataset, das einen xarray-Batch in einen normalisierten Tensor konvertiert."""

    def __init__(self, bgen: xb.BatchGenerator):
        super().__init__()

        self.bgen = bgen

        dask.config.set(scheduler="threads", num_workers=4)

        mean = np.array([2.76707305e+02, -1.02550652e-01, -8.24716593e-02,  1.01068682e+05,
                         2.13901834e+02,  2.09669021e+02,  2.14224057e+02,  2.18012186e+02,
                         2.22117960e+02,  2.27618180e+02,  2.40553091e+02,  2.51450718e+02,
                         2.59819244e+02,  2.66263193e+02,  2.73431732e+02,  2.76170563e+02,
                         2.79528167e+02,  3.87254235e+00,  9.39696721e+00,  1.39809760e+01,
                         1.49588660e+01,  1.42096134e+01,  1.26746424e+01,  9.40749201e+00,
                         6.76743938e+00,  4.85057830e+00,  3.21840022e+00,  1.19613039e+00,
                         3.40955153e-01, -2.00982027e-01,  1.34509794e-01,  1.86537438e-02,
                         1.77366811e-01,  2.60285472e-01,  1.08158604e-01,  2.13348037e-02,
                         -5.33151006e-02, -1.12940699e-02,  1.37653121e-02,  1.64470187e-02,
                         -5.36961093e-03, -1.42718665e-02, -8.16306830e-02,  1.99295885e+05,
                         1.57330177e+05,  1.32683094e+05,  1.14840669e+05,  1.00754974e+05,
                         8.89962866e+04,  6.96855338e+04,  5.39212137e+04,  4.05297225e+04,
                         2.88684465e+04,  1.37619912e+04,  7.06023469e+03,  8.15529195e+02,
                         2.87899168e-06,  2.44946302e-06,  4.41716612e-06,  1.54408574e-05,
                         4.63313069e-05,  1.05735979e-04,  3.32204274e-04,  7.38973747e-04,
                         1.37365580e-03,  2.20929030e-03,  4.23163159e-03,  5.59333540e-03,
                         6.48287372e-03])

        std = np.array([2.09942404e+01, 5.25000636e+00, 4.54455487e+00, 1.30960034e+03,
                        8.97812032e+00, 1.32565826e+01, 8.31339312e+00, 5.15994231e+00,
                        6.88576031e+00, 9.93203450e+00, 1.24352490e+01, 1.29195538e+01,
                        1.30728671e+01, 1.40098769e+01, 1.47487644e+01, 1.53852921e+01,
                        1.71116930e+01, 1.00916061e+01, 1.18567912e+01, 1.51044572e+01,
                        1.70482496e+01, 1.72106285e+01, 1.64754925e+01, 1.39160706e+01,
                        1.17258202e+01, 1.00555255e+01, 8.94536813e+00, 7.80402390e+00,
                        7.49754381e+00, 5.91365735e+00, 7.13226032e+00, 7.68995984e+00,
                        9.47791003e+00, 1.15089522e+01, 1.27980862e+01, 1.27539256e+01,
                        1.08107437e+01, 8.95480061e+00, 7.69034815e+00, 6.91974370e+00,
                        6.33759832e+00, 6.47175201e+00, 5.22074238e+00, 2.97759049e+03,
                        3.99086247e+03, 4.97500846e+03, 5.28610563e+03, 5.15772933e+03,
                        4.77762842e+03, 3.87501782e+03, 3.09197738e+03, 2.45088338e+03,
                        1.91940426e+03, 1.30757654e+03, 1.10889327e+03, 1.01593943e+03,
                        1.87661911e-07, 4.75091686e-07, 2.64407777e-06, 1.75901199e-05,
                        6.03882715e-05, 1.42577167e-04, 4.54680063e-04, 9.75985021e-04,
                        1.64347251e-03, 2.37802664e-03, 3.98016829e-03, 4.98595989e-03,
                        5.80280740e-03])

        self.normalizer = Normalize(
            mean=mean,
            std=std,
            inplace=True
        )

    def __len__(self):
        return len(self.bgen)

    def __getitem__(self, idx):
        batch = self.bgen[idx]

        stacked = batch.to_stacked_array(
            new_dim="channel",
            sample_dims=["latitude", "longitude"]
        ).transpose("channel", "longitude", "latitude")

        x = torch.tensor(stacked.data)

        x = x[:, :, :-1]

        x = self.normalizer(x)

        return x


class ERA5DataModule(pl.LightningDataModule):
    """Lightning DataModule für ERA5-Training/Validierung."""

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        bgen_train = gen_bgen(train=True)
        bgen_val = gen_bgen(train=False)

        self.train_ds = ERA5Dataset(bgen=bgen_train)
        self.val_ds = ERA5Dataset(bgen=bgen_val)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config["config"]["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["config"]["dataloader"]["num_workers"],
            persistent_workers=True,
            multiprocessing_context="spawn"
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.config["config"]["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["config"]["dataloader"]["num_workers"],
            persistent_workers=True,
            multiprocessing_context="spawn"
        )


def main():

    import yaml

    with open("configs/autoencoder_kl_f8.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data = ERA5DataModule(config)

    count = 0

    for x in data.train_dataloader():

        print(x.shape)

        count += 1

        if count == 100:
            break


if __name__ == "__main__":
    main()
