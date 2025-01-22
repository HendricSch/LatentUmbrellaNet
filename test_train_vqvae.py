import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

from models.vqvae import VQVAE
from dataset.vae_datamodule import VAEDataModule


def train():

    torch.set_float32_matmul_precision("medium")

    model = VQVAE(in_channels=5)
    # model = VQVAE.load_from_checkpoint("logs/vqvae_5channel_full_ds_64_128_256_512_c1024/version_0/checkpoints/epoch=5-step=27111.ckpt", in_channels=5)

    dm = VAEDataModule(
        train_path="train.memmap",
        val_path="val.memmap",
        test_path="test.memmap",
        img_channel=5,
        img_res=(128, 64),
        batch_size=64,
        num_workers=0,
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="logs/", name="vqvae_5channel_test")

    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=-1,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
