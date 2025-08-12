import torch
import torch.nn as nn
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
import yaml

from data.era5 import ERA5DataModule
from models.autoencoder import Autoencoder


def main():
    torch.set_float32_matmul_precision("medium")

    with open("configs/autoencoder/kl-f8-disc.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data = ERA5DataModule(config)

    # autoencoder = Autoencoder(config)
    autoencoder = Autoencoder.load_from_checkpoint(
        "checkpoints/vae-kl-f8-rmse.ckpt", config=config, strict=False
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=config["config"]["general"]["name"] + "-{step}",
        every_n_train_steps=2500,
    )

    trainer = lightning.Trainer(
        max_epochs=config["config"]["training"]["epochs"],
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        limit_val_batches=500,
        val_check_interval=2500,
    )

    trainer.fit(autoencoder, datamodule=data)


if __name__ == "__main__":
    main()
