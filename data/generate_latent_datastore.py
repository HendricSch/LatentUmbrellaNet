import dask.config
import xarray as xr
import xbatcher as xb
import numpy as np
import dask
import dask.array as da
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from multiprocessing import Pool as ThreadPool
from torchvision.transforms import Normalize
import yaml

from models.autoencoder import Autoencoder


@torch.no_grad()
def normalizer(x: torch.Tensor) -> torch.Tensor:
    mean = np.array(
        [
            2.76707305e02,
            -1.02550652e-01,
            -8.24716593e-02,
            1.01068682e05,
            2.13901834e02,
            2.09669021e02,
            2.14224057e02,
            2.18012186e02,
            2.22117960e02,
            2.27618180e02,
            2.40553091e02,
            2.51450718e02,
            2.59819244e02,
            2.66263193e02,
            2.73431732e02,
            2.76170563e02,
            2.79528167e02,
            3.87254235e00,
            9.39696721e00,
            1.39809760e01,
            1.49588660e01,
            1.42096134e01,
            1.26746424e01,
            9.40749201e00,
            6.76743938e00,
            4.85057830e00,
            3.21840022e00,
            1.19613039e00,
            3.40955153e-01,
            -2.00982027e-01,
            1.34509794e-01,
            1.86537438e-02,
            1.77366811e-01,
            2.60285472e-01,
            1.08158604e-01,
            2.13348037e-02,
            -5.33151006e-02,
            -1.12940699e-02,
            1.37653121e-02,
            1.64470187e-02,
            -5.36961093e-03,
            -1.42718665e-02,
            -8.16306830e-02,
            1.99295885e05,
            1.57330177e05,
            1.32683094e05,
            1.14840669e05,
            1.00754974e05,
            8.89962866e04,
            6.96855338e04,
            5.39212137e04,
            4.05297225e04,
            2.88684465e04,
            1.37619912e04,
            7.06023469e03,
            8.15529195e02,
            2.87899168e-06,
            2.44946302e-06,
            4.41716612e-06,
            1.54408574e-05,
            4.63313069e-05,
            1.05735979e-04,
            3.32204274e-04,
            7.38973747e-04,
            1.37365580e-03,
            2.20929030e-03,
            4.23163159e-03,
            5.59333540e-03,
            6.48287372e-03,
        ]
    )

    std = np.array(
        [
            2.09942404e01,
            5.25000636e00,
            4.54455487e00,
            1.30960034e03,
            8.97812032e00,
            1.32565826e01,
            8.31339312e00,
            5.15994231e00,
            6.88576031e00,
            9.93203450e00,
            1.24352490e01,
            1.29195538e01,
            1.30728671e01,
            1.40098769e01,
            1.47487644e01,
            1.53852921e01,
            1.71116930e01,
            1.00916061e01,
            1.18567912e01,
            1.51044572e01,
            1.70482496e01,
            1.72106285e01,
            1.64754925e01,
            1.39160706e01,
            1.17258202e01,
            1.00555255e01,
            8.94536813e00,
            7.80402390e00,
            7.49754381e00,
            5.91365735e00,
            7.13226032e00,
            7.68995984e00,
            9.47791003e00,
            1.15089522e01,
            1.27980862e01,
            1.27539256e01,
            1.08107437e01,
            8.95480061e00,
            7.69034815e00,
            6.91974370e00,
            6.33759832e00,
            6.47175201e00,
            5.22074238e00,
            2.97759049e03,
            3.99086247e03,
            4.97500846e03,
            5.28610563e03,
            5.15772933e03,
            4.77762842e03,
            3.87501782e03,
            3.09197738e03,
            2.45088338e03,
            1.91940426e03,
            1.30757654e03,
            1.10889327e03,
            1.01593943e03,
            1.87661911e-07,
            4.75091686e-07,
            2.64407777e-06,
            1.75901199e-05,
            6.03882715e-05,
            1.42577167e-04,
            4.54680063e-04,
            9.75985021e-04,
            1.64347251e-03,
            2.37802664e-03,
            3.98016829e-03,
            4.98595989e-03,
            5.80280740e-03,
        ]
    )

    normalizer = Normalize(mean=mean, std=std, inplace=True)

    x = normalizer(x)

    return x


@torch.no_grad()
def main():
    # Konstanten
    AUTOENCODER_CONFIG = "configs/autoencoder/kl-f8.yaml"
    AUTOENCODER_CHECKPOINT = "checkpoints/vae-kl-f8-rmse-disc-2-step=5000-z500=93.ckpt"
    LATENT_STORE_PATH = "zarr_files/latent_moments_vae-kl-f8.zarr"
    TIME_START = "1940-01-01"
    NUM_SAMPLES = 25088
    NUM_THREADS = 64
    THREAD_CHUNK_SIZE = 128
    DEVICE = "cuda"

    assert NUM_SAMPLES % THREAD_CHUNK_SIZE == 0, (
        f"NUM_SAMPLES ({NUM_SAMPLES}) must be divisible by THREAD_CHUNK_SIZE ({THREAD_CHUNK_SIZE})"
    )

    assert THREAD_CHUNK_SIZE % NUM_THREADS == 0, (
        f"THREAD_CHUNK_SIZE ({THREAD_CHUNK_SIZE}) must be divisible by NUM_THREADS ({NUM_THREADS})"
    )

    # Initialisiere den VAE (nur Encoder wird benötigt)
    with open(AUTOENCODER_CONFIG, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    autoencoder = Autoencoder.load_from_checkpoint(
        AUTOENCODER_CHECKPOINT, config=config, strict=False
    ).to(DEVICE)
    autoencoder.eval()
    autoencoder.freeze()

    # Erstelle Zarr-Store für die latenten Wetterdaten
    x_cord = np.arange(0, 180, 1)
    y_cord = np.arange(0, 90, 1)
    channel_cord = np.arange(0, 128, 1)
    time_cord = pd.date_range(TIME_START, periods=NUM_SAMPLES, freq="6h")

    dataset = xr.Dataset(
        coords={
            "time": (["time"], time_cord),
            "x": (["x"], x_cord),
            "y": (["y"], y_cord),
            "channel": (["channel"], channel_cord),
        },
        data_vars={
            "data": (
                ["time", "channel", "x", "y"],
                da.zeros(
                    (time_cord.size, channel_cord.size, x_cord.size, y_cord.size),
                    chunks=(1, 128, 180, 90),
                    dtype="float16",
                ),
            )
        },
    )

    dataset.to_zarr(
        LATENT_STORE_PATH,
        compute=False,
        mode="w",
        consolidated=False,
        encoding={"data": {"dtype": "float16"}},
    )

    # Initialisiere ERA5 Dataset aus der Cloud
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=None,
        storage_options=dict(token="anon"),
    )

    # Erstelle den BatchGenerator
    LEVEL = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    VARS_SURFACE = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
    ]
    VARS_ATMOSPHERE = [
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential",
        "specific_humidity",
    ]

    ds = ds.sel(time=time_cord)
    ds = ds.sel(level=LEVEL)
    ds = ds[VARS_SURFACE + VARS_ATMOSPHERE]

    bgen = xb.BatchGenerator(
        ds, input_dims={"time": 1, "level": 13, "latitude": 721, "longitude": 1440}
    )

    # Iteriere über alle Wetterdaten innerhalb eines Zeitraumes
    # Lade jeweils ein Batch, berechne die latente Repräsentation und speicher diese im Zarr-Store

    pool = ThreadPool(NUM_THREADS)
    job = pool.map_async(bgen.__getitem__, range(0, THREAD_CHUNK_SIZE))
    batches: list[xr.Dataset] = None

    for i in tqdm(range(NUM_SAMPLES // THREAD_CHUNK_SIZE), position=0, leave=True):
        batches = job.get()

        if i != (NUM_SAMPLES // THREAD_CHUNK_SIZE) - 1:
            job = pool.map_async(
                bgen.__getitem__,
                range((i + 1) * THREAD_CHUNK_SIZE, (i + 2) * THREAD_CHUNK_SIZE),
            )

        for batch in tqdm(batches, position=1, leave=False):
            stacked = batch.to_stacked_array(
                new_dim="channel", sample_dims=["latitude", "longitude"]
            ).transpose("channel", "longitude", "latitude")

            x = torch.tensor(stacked.data)

            x = x[:, :, :-1]

            x = normalizer(x)

            x = x.unsqueeze(0)
            x = x.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                # posterior = autoencoder.encode(x)
                # z = posterior.sample()
                # z = z.cpu().numpy()
                h = autoencoder.encoder.forward(x)
                z = autoencoder.quant_conv.forward(h)
                z = z.type(torch.float16)
                z = z.cpu().numpy()

            date = batch.time.values[0]

            new_data = xr.Dataset(
                coords={
                    "time": (["time"], [date]),
                    "x": (["x"], x_cord),
                    "y": (["y"], y_cord),
                    "channel": (["channel"], channel_cord),
                },
                data_vars={
                    "data": (
                        ["time", "channel", "x", "y"],
                        z,
                    )
                },
            )

            new_data.to_zarr(LATENT_STORE_PATH, region="auto", consolidated=False)
            new_data.close()

    pool.close()


if __name__ == "__main__":
    main()
