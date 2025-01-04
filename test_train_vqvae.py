import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

from models.vqvae import VQVAE
from dataset.vae_datamodule import VAEDataModule


def main():

    torch.set_float32_matmul_precision("medium")

    model = VQVAE(
        in_channels=5
    )
    dm = VAEDataModule(
        train_path="train.memmap",
        val_path="val.memmap",
        test_path="test.memmap",
        img_channel=5,
        img_res=(128, 64),
        batch_size=64,
        num_workers=1,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name="vqvae_5channel")

    trainer = pl.Trainer(
            logger=tb_logger,
            check_val_every_n_epoch=5,
            max_epochs=-1,
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    main()