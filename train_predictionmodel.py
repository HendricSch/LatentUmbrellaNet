import torch
import torch.nn as nn
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from data.latents import LatentsDataModule
from models.predictionnet import PredictionModel, AFNOPredictionModel


def main():
    torch.set_float32_matmul_precision("medium")

    data = LatentsDataModule(batch_size=2)

    model = PredictionModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="prediction-model-{val_loss:.5f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = lightning.Trainer(
        max_epochs=1, precision="16-mixed", callbacks=[checkpoint_callback, lr_monitor]
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
