from models.vae import VAE
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

from dataset.geopotential_vae import GeopotentialDataModuleVAE


def main():
    model = VAE()
    dm = GeopotentialDataModuleVAE(batch_size=32)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name="vae")

    trainer = pl.Trainer(
        max_epochs=20,
        precision=16,
        limit_train_batches=0.1,
        limit_val_batches=0.01,
        logger=tb_logger,
    )
    # trainer = pl.Trainer(
    #     max_epochs=20,
    #     precision=16,
    #     logger=tb_logger,
    # )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
