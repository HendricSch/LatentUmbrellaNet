import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import OneCycleLR

from models.vqvae_2 import VQVAE2
from models.discriminator import PatchGanDiscriminator


class VQVAE2FineTune(pl.LightningModule):

    def __init__(self, vqvae: VQVAE2, lr: float) -> None:
        super(VQVAE2FineTune, self).__init__()

        # Hyperparameters Lightining
        self.lr = lr
        self.example_input_array = torch.rand(1, 5, 128, 64)
        self.automatic_optimization = False

        # Hyperparameters VQVAE
        self.vqvae = vqvae

        # freeze VQVAE parameters except the decoder
        for name, param in vqvae.named_parameters():
            if "decoder" not in name:
                param.requires_grad = False

        # Discriminator
        self.discriminator = PatchGanDiscriminator(in_channels=5)
        self.disc_loss_weight = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vqvae(x)

    def training_step(self, batch, batch_idx):
        g_optimizer, d_optimizer = self.optimizers()

        x = batch
        pred, _, _ = self(x)

        ### Optimize Discriminator ###

        disc_fake_pred = self.discriminator(pred.detach())
        disc_real_pred = self.discriminator(x)

        disc_fake_loss = F.mse_loss(
            disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = F.mse_loss(
            disc_real_pred, torch.ones_like(disc_real_pred))

        disc_loss = (disc_fake_loss + disc_real_loss) * 0.5

        d_optimizer.zero_grad()
        self.manual_backward(disc_loss)
        d_optimizer.step()

        ### Optimize Generator ###

        disc_fake_pred = self.discriminator(pred)

        recon_loss = F.mse_loss(pred, x)

        gen_loss = self.disc_loss_weight * \
            F.mse_loss(disc_fake_pred, torch.ones_like(
                disc_fake_pred)) + recon_loss

        g_optimizer.zero_grad()
        self.manual_backward(gen_loss)
        g_optimizer.step()

        self.log_dict({"train_gen_loss": gen_loss,
                       "train_disc_loss": disc_loss, "train_recon_loss": recon_loss})

        # log an image of the reconstruction every 100 steps to tensorboard
        if batch_idx % 100 == 0:
            sample_plot = x[0:3, 0, :, :].unsqueeze(1)
            prediction_plot = pred[0:3, 0, :, :].unsqueeze(1)
            self.logger.experiment.add_image(
                "c500",
                torchvision.utils.make_grid(
                    torch.cat([sample_plot, prediction_plot], dim=0)
                ),
                self.current_epoch,
            )

            sample_plot = x[0:3, 3, :, :].unsqueeze(1)
            prediction_plot = pred[0:3, 3, :, :].unsqueeze(1)
            self.logger.experiment.add_image(
                "t2m",
                torchvision.utils.make_grid(
                    torch.cat([sample_plot, prediction_plot], dim=0)
                ),
                self.current_epoch,
            )

    def validation_step(self, batch, batch_idx):
        x = batch
        pred, _, _ = self(x)

        ### Optimize Discriminator ###

        disc_fake_pred = self.discriminator(pred.detach())
        disc_real_pred = self.discriminator(x)

        disc_fake_loss = F.mse_loss(
            disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = F.mse_loss(
            disc_real_pred, torch.ones_like(disc_real_pred))

        disc_loss = (disc_fake_loss + disc_real_loss) * 0.5

        ### Optimize Generator ###

        disc_fake_pred = self.discriminator(pred)

        recon_loss = F.mse_loss(pred, x)

        gen_loss = self.disc_loss_weight * \
            F.mse_loss(disc_fake_pred, torch.ones_like(
                disc_fake_pred)) + recon_loss

        self.log_dict({"val_gen_loss": gen_loss, "val_disc_loss": disc_loss})

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.vqvae.parameters(), lr=self.lr)
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr)

        return [optimizer_g, optimizer_d]
