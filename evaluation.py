import pandas as pd
import numpy as np
from multiprocessing import Pool as ThreadPool
from tqdm import tqdm
import dask.config
import xarray as xr
import xbatcher as xb
import numpy as np
import dask
import torch
import random
import os

from data.era5 import gen_bgen
from metrics.metrics import WeightedRMSE
from models.latent_umbrella_net import LatentUmbrellaNet
from models.autoencoder import Autoencoder

NUM_WORKERS = 6


def eval_persistence(
    forecast_steps: int = 3, rounds: int = 1, save_to_csv: bool = True
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    for i in tqdm(range(rounds)):
        # create a thread pool for parallel processing
        pool = ThreadPool(NUM_WORKERS)

        # create a batch generator for the era5 data
        bgen = gen_bgen(train=True)

        # seed
        s = random.randint(0, 1000)

        indexes = np.arange(0, 6 * forecast_steps, 6) + s
        indexes = indexes.tolist()

        # load the data parallelly from the gsc
        job = pool.map_async(bgen.__getitem__, indexes)
        batches: list[xr.Dataset] = job.get()
        pool.close()

        # convert the batches to torch tensors
        data = []
        for batch in batches:
            stacked = batch.to_stacked_array(
                new_dim="channel", sample_dims=["latitude", "longitude"]
            ).transpose("channel", "longitude", "latitude")

            item = torch.tensor(stacked.data)
            item = item.unsqueeze(0)
            item = item[:, :, :, :-1]

            data.append(item)

        data = torch.cat(data, dim=0)

        wrmse = WeightedRMSE(num_latitudes=720)

        persistence = np.array([wrmse(data[0].numpy(), item.numpy()) for item in data])

        persistence_dict = {
            "z500": persistence[:, 50],
            "t850": persistence[:, 14],
            "h700": persistence[:, 65],
            "t2m": persistence[:, 0],
            "u10": persistence[:, 1],
            "u850": persistence[:, 27],
        }

        df = pd.DataFrame(persistence_dict)
        dfs.append(df)

    # mean the dfs
    res_df = sum(dfs) / len(dfs)

    if save_to_csv:
        if not os.path.exists("./evaluation"):
            os.makedirs("./evaluation")

        res_df.to_csv("evaluation/persistence.csv", index=False, header=True, mode="w")

    return res_df


def eval_climatology(
    sample_size: int = 5, rounds: int = 1, save_to_csv: bool = True
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    for i in range(rounds):
        # create a thread pool for parallel processing
        pool = ThreadPool(NUM_WORKERS)

        # create a batch generator for the era5 data
        bgen = gen_bgen(train=True)

        indexes = [random.randint(0, 1000) for _ in range(sample_size)]

        # load the data parallelly from the gsc
        job = pool.map_async(bgen.__getitem__, indexes)
        batches: list[xr.Dataset] = job.get()
        pool.close()

        # convert the batches to torch tensors
        data = []
        for batch in batches:
            stacked = batch.to_stacked_array(
                new_dim="channel", sample_dims=["latitude", "longitude"]
            ).transpose("channel", "longitude", "latitude")

            item = torch.tensor(stacked.data)
            item = item.unsqueeze(0)
            item = item[:, :, :, :-1]

            data.append(item)

        data = torch.cat(data, dim=0)

        mean_sample = data.mean(dim=0)

        wrmse = WeightedRMSE(num_latitudes=720)

        climatology = wrmse(mean_sample.numpy(), data[0].numpy())

        climatology_dict = {
            "z500": climatology[50],
            "t850": climatology[14],
            "h700": climatology[65],
            "t2m": climatology[0],
            "u10": climatology[1],
            "u850": climatology[27],
        }

        df = pd.DataFrame(climatology_dict, index=[0])
        dfs.append(df)

    res_df = sum(dfs) / len(dfs)

    if save_to_csv:
        if not os.path.exists("./evaluation"):
            os.makedirs("./evaluation")

        res_df.to_csv("evaluation/climatology.csv", index=False, header=True, mode="w")

    return res_df


def eval_vae_error(rounds: int = 1, save_to_csv: bool = True) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    lun = LatentUmbrellaNet(
        vae_ckpt_path="checkpoints/vae-kl-f8-rmse-disc-2-step=5000-z500=93.ckpt",
        vae_config_path="configs/autoencoder/kl-f8-disc.yaml",
        prediction_net_ckpt_path="checkpoints/prediction-model-val_loss=0.01241.ckpt",
        device="cuda",
        prediction_net_type="unet",
    )

    for i in range(rounds):
        bgen = gen_bgen(train=True)

        index = random.randint(0, 1000)

        batch = bgen[index]

        stacked = batch.to_stacked_array(
            new_dim="channel", sample_dims=["latitude", "longitude"]
        ).transpose("channel", "longitude", "latitude")

        item = torch.tensor(stacked.data)
        item = item.unsqueeze(0)
        item = item[:, :, :, :-1]

        recon = lun.encode_decode(item)

        wrmse = WeightedRMSE(num_latitudes=720)

        vae_error = wrmse(recon.numpy(), item.numpy())

        vae_error_dict = {
            "z500": vae_error[:, 50],
            "t850": vae_error[:, 14],
            "h700": vae_error[:, 65],
            "t2m": vae_error[:, 0],
            "u10": vae_error[:, 1],
            "u850": vae_error[:, 27],
        }

        df = pd.DataFrame(vae_error_dict, index=[0])

        dfs.append(df)

    res_df = sum(dfs) / len(dfs)

    if save_to_csv:
        if not os.path.exists("./evaluation"):
            os.makedirs("./evaluation")

        res_df.to_csv("evaluation/vae_error.csv", index=False, header=True, mode="w")

    return res_df


def eval_lun_unet(
    forecast_steps: int = 4, rounds: int = 1, save_to_csv: bool = True
) -> pd.DataFrame:
    lun = LatentUmbrellaNet(
        vae_ckpt_path="checkpoints/vae-kl-f8-rmse-disc-2-step=5000-z500=93.ckpt",
        vae_config_path="configs/autoencoder/kl-f8-disc.yaml",
        prediction_net_ckpt_path="checkpoints/prediction-model-val_loss=0.01241.ckpt",
        device="cuda",
        prediction_net_type="unet",
    )

    dfs: list[pd.DataFrame] = []

    for _ in range(rounds):
        # create a thread pool for parallel processing
        pool = ThreadPool(NUM_WORKERS)

        # create a batch generator for the era5 data
        bgen = gen_bgen(train=True)

        # seed
        s = random.randint(0, 1000)

        indexes = np.arange(0, 6 * (forecast_steps + 2), 6) + s
        indexes = indexes.tolist()

        # load the data parallelly from the gsc
        job = pool.map_async(bgen.__getitem__, indexes)
        batches: list[xr.Dataset] = job.get()
        pool.close()

        # convert the batches to torch tensors
        data = []
        for batch in batches:
            stacked = batch.to_stacked_array(
                new_dim="channel", sample_dims=["latitude", "longitude"]
            ).transpose("channel", "longitude", "latitude")

            item = torch.tensor(stacked.data)
            item = item.unsqueeze(0)
            item = item[:, :, :, :-1]

            data.append(item)

        data = torch.cat(data, dim=0)

        forecastst = []

        for i in range(forecast_steps):
            forecast = lun.forward(data[0].unsqueeze(0), data[1].unsqueeze(0), i + 1)
            forecastst.append(forecast)

        data = data[2:]

        forecastst = torch.cat(forecastst, dim=0)

        wrmse = WeightedRMSE(num_latitudes=720)

        lun_unet = np.array(
            [
                wrmse(data[i].numpy(), forecastst[i].numpy())
                for i in range(forecast_steps)
            ]
        )

        lun_unet_dict = {
            "z500": lun_unet[:, 50],
            "t850": lun_unet[:, 14],
            "h700": lun_unet[:, 65],
            "t2m": lun_unet[:, 0],
            "u10": lun_unet[:, 1],
            "u850": lun_unet[:, 27],
        }

        df = pd.DataFrame(lun_unet_dict)
        dfs.append(df)

    res_df = sum(dfs) / len(dfs)

    if save_to_csv:
        if not os.path.exists("./evaluation"):
            os.makedirs("./evaluation")

        res_df.to_csv("evaluation/lun_unet.csv", index=False, header=True, mode="w")

    return res_df


def eval_lun_fourcastnet(
    forecast_steps: int = 4, rounds: int = 1, save_to_csv: bool = True
) -> pd.DataFrame:
    lun = LatentUmbrellaNet(
        vae_ckpt_path="checkpoints/vae-kl-f8-rmse-disc-2-step=5000-z500=93.ckpt",
        vae_config_path="configs/autoencoder/kl-f8-disc.yaml",
        prediction_net_ckpt_path="checkpoints/prediction-model-afno-val_loss=0.01546.ckpt",
        device="cuda",
        prediction_net_type="afno",
    )

    dfs: list[pd.DataFrame] = []

    for _ in range(rounds):
        # create a thread pool for parallel processing
        pool = ThreadPool(NUM_WORKERS)

        # create a batch generator for the era5 data
        bgen = gen_bgen(train=True)

        # seed
        s = random.randint(0, 1000)

        indexes = np.arange(0, 6 * (forecast_steps + 2), 6) + s
        indexes = indexes.tolist()

        # load the data parallelly from the gsc
        job = pool.map_async(bgen.__getitem__, indexes)
        batches: list[xr.Dataset] = job.get()
        pool.close()

        # convert the batches to torch tensors
        data = []
        for batch in batches:
            stacked = batch.to_stacked_array(
                new_dim="channel", sample_dims=["latitude", "longitude"]
            ).transpose("channel", "longitude", "latitude")

            item = torch.tensor(stacked.data)
            item = item.unsqueeze(0)
            item = item[:, :, :, :-1]

            data.append(item)

        data = torch.cat(data, dim=0)

        forecastst = []

        for i in range(forecast_steps):
            forecast = lun.forward(data[0].unsqueeze(0), data[1].unsqueeze(0), i + 1)
            forecastst.append(forecast)

        data = data[2:]

        forecastst = torch.cat(forecastst, dim=0)

        wrmse = WeightedRMSE(num_latitudes=720)

        lun_unet = np.array(
            [
                wrmse(data[i].numpy(), forecastst[i].numpy())
                for i in range(forecast_steps)
            ]
        )

        lun_unet_dict = {
            "z500": lun_unet[:, 50],
            "t850": lun_unet[:, 14],
            "h700": lun_unet[:, 65],
            "t2m": lun_unet[:, 0],
            "u10": lun_unet[:, 1],
            "u850": lun_unet[:, 27],
        }

        df = pd.DataFrame(lun_unet_dict)
        dfs.append(df)

    res_df = sum(dfs) / len(dfs)

    if save_to_csv:
        if not os.path.exists("./evaluation"):
            os.makedirs("./evaluation")

        res_df.to_csv(
            "evaluation/lun_fourcastnet.csv", index=False, header=True, mode="w"
        )

    return res_df


if __name__ == "__main__":
    # eval_persistence()
    # eval_climatology()
    # eval_vae_error()
    # eval_lun_unet()
    eval_lun_fourcastnet(forecast_steps=2, rounds=3, save_to_csv=True)
