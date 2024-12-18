from models.vqvae import VQVAE
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

from dataset.geopotential_vae import GeopotentialDataModuleVAE


def main():
    model = VQVAE()
    dm = GeopotentialDataModuleVAE(batch_size=4)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name="vqvae")

    trainer = pl.Trainer(
        max_epochs=100,
        limit_train_batches=0.1,
        limit_val_batches=0.01,
        logger=tb_logger,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
