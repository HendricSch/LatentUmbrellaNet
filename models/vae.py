import torch
import torch.nn as nn
import lightning.pytorch as pl
import numpy as np
import torch.nn.functional as F
import torchvision

from models.blocks import DownBlock, MidBlock, UpBlock
from models.discriminator import Discriminator


class VAE(pl.LightningModule):
    def __init__(self):
        super(VAE, self).__init__()

        # Hyperparameters Lightining
        self.lr = 0.0001
        self.example_input_array = torch.rand(1, 1, 128, 64)
        self.automatic_optimization = False

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.step_count = 0

        # Hyperparameters VAE
        self.latent_dim = 4
        self.kl_weight = 0.000005

        # Discriminator
        self.discriminator = Discriminator(in_channels=1)
        self.disc_weight = 0.5

        ### Encoder ###
        self.encoder_conv_in = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # DownBlocks + MidBlock

        self.encoder_down1 = DownBlock(32, 64)
        self.encoder_down2 = DownBlock(64, 128)
        self.encoder_down3 = DownBlock(128, 128)

        self.encoder_mid_1 = MidBlock(128, 128)
        self.encoder_mid_2 = MidBlock(128, 128)

        self.encoder_norm_out = nn.GroupNorm(32, 128)
        self.encoder_conv_out = nn.Conv2d(
            128, 2 * self.latent_dim, kernel_size=3, padding=1
        )
        self.pre_quant_conv = nn.Conv2d(
            2 * self.latent_dim, 2 * self.latent_dim, kernel_size=1
        )

        ### Decoder ###
        self.post_quant_conv = nn.Conv2d(
            self.latent_dim, self.latent_dim, kernel_size=1
        )

        self.decoder_conv_in = nn.Conv2d(self.latent_dim, 128, kernel_size=3, padding=1)

        # MidBlock + UpBlocks

        self.decoder_mid_1 = MidBlock(128, 128)
        self.decoder_mid_2 = MidBlock(128, 128)

        self.decoder_up1 = UpBlock(128, 128)
        self.decoder_up2 = UpBlock(128, 64)
        self.decoder_up3 = UpBlock(64, 32)

        self.decoder_norm_out = nn.GroupNorm(32, 32)
        self.decoder_conv_out = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def encode(self, x):
        out = self.encoder_conv_in(x)

        out = self.encoder_down1(out)
        out = self.encoder_down2(out)
        # out = self.encoder_down3(out)

        out = self.encoder_mid_1(out)
        out = self.encoder_mid_2(out)

        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)

        mean, logvar = torch.chunk(out, 2, dim=1)

        std = torch.exp(0.5 * logvar)

        sample = mean + std * torch.randn(mean.shape).to(device=x.device)

        return sample, out

    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)

        out = self.decoder_mid_1(out)
        out = self.decoder_mid_2(out)

        # out = self.decoder_up1(out)
        out = self.decoder_up2(out)
        out = self.decoder_up3(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)

        return out

    def forward(self, x):
        z, encoder_out = self.encode(x)
        out = self.decode(z)
        return out, encoder_out

    def calc_losses_discriminator(self, sample, prediction) -> dict:
        disc_fake_pred = self.discriminator(prediction.detach())
        disc_real_pred = self.discriminator(sample)

        disc_fake_loss = F.mse_loss(
            disc_fake_pred,
            torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device),
        )

        disc_real_loss = F.mse_loss(
            disc_real_pred,
            torch.ones(disc_real_pred.shape, device=disc_real_pred.device),
        )

        disc_loss = self.disc_weight * (disc_fake_loss + disc_real_loss) / 2

        return disc_loss

    def calc_losses_generator(self, sample, prediction, encoder_out) -> dict:
        # Reconstruction loss
        recon_loss = F.mse_loss(prediction, sample)

        # KL divergence loss
        mean, logvar = torch.chunk(encoder_out, 2, dim=1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss * self.kl_weight

        # Adversarial loss
        disc_fake_pred = self.discriminator(prediction)
        adversarial_loss = F.mse_loss(
            disc_fake_pred,
            torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device),
        )
        adversarial_loss = self.disc_weight * adversarial_loss

        loss_dict = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "adversarial_loss": adversarial_loss,
        }

        return loss_dict

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()

        sample = batch
        prediction, encoder_out = self(sample)

        # Calculate losses for generator
        loss_generator_dict = self.calc_losses_generator(
            sample, prediction, encoder_out
        )
        loss_generator_kl = loss_generator_dict["kl_loss"]
        loss_generator_recon = loss_generator_dict["recon_loss"]
        loss_generator_adversarial = loss_generator_dict["adversarial_loss"]

        if self.step_count > 1000:
            loss_generator = (
                loss_generator_kl + loss_generator_recon + loss_generator_adversarial
            )
        else:
            loss_generator = loss_generator_recon + loss_generator_kl

        # Train generator
        optimizer_g.zero_grad()
        self.manual_backward(loss_generator)
        optimizer_g.step()

        # Calculate losses for discriminator
        loss_discriminator = self.calc_losses_discriminator(sample, prediction)

        if self.step_count > 1000:
            # Train discriminator
            optimizer_d.zero_grad()
            self.manual_backward(loss_discriminator)
            optimizer_d.step()

        self.training_step_outputs.append(loss_generator.item())

        self.log_dict(
            {
                "train_recon_loss": loss_generator_recon,
                "train_kl_loss": loss_generator_kl,
                "train_adversarial_loss": loss_generator_adversarial,
                "train_generator_loss": loss_generator,
                "train_disc_loss": loss_discriminator,
            }
        )

        self.step_count += 1

        # log an image of the reconstruction every 100 steps to tensorboard
        if batch_idx % 100 == 0:
            self.logger.experiment.add_image(
                "reconstruction",
                torchvision.utils.make_grid(
                    torch.cat([sample[0:3], prediction[0:3]], dim=0)
                ),
                self.current_epoch,
            )

    def on_train_epoch_end(self):
        epoch_mean = np.mean(self.training_step_outputs)
        self.log("training_epoch_loss", epoch_mean, on_step=False, on_epoch=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        sample = batch
        prediction, encoder_out = self(sample)

        loss_generator_dict = self.calc_losses_generator(
            sample, prediction, encoder_out
        )

        loss_generator_kl = loss_generator_dict["kl_loss"]
        loss_generator_recon = loss_generator_dict["recon_loss"]
        loss_generator_adversarial = loss_generator_dict["adversarial_loss"]
        loss_generator = (
            loss_generator_kl + loss_generator_recon + loss_generator_adversarial
        )

        self.validation_step_outputs.append(loss_generator.item())

        self.log_dict(
            {
                "val_recon_loss": loss_generator_recon,
                "val_kl_loss": loss_generator_kl,
                "val_adversarial_loss": loss_generator_adversarial,
                "val_loss": loss_generator,
            }
        )

    def on_validation_epoch_end(self):
        epoch_mean = np.mean(self.validation_step_outputs)
        self.log("validation_epoch_loss", epoch_mean, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        sample = batch
        prediction, encoder_out = self(sample)

        loss_generator_dict = self.calc_losses_generator(
            sample, prediction, encoder_out
        )

        loss_generator_kl = loss_generator_dict["kl_loss"]
        loss_generator_recon = loss_generator_dict["recon_loss"]
        loss_generator_adversarial = loss_generator_dict["adversarial_loss"]
        loss_generator = (
            loss_generator_kl + loss_generator_recon + loss_generator_adversarial
        )

        self.validation_step_outputs.append(loss_generator.item())

        self.log_dict(
            {
                "test_recon_loss": loss_generator_recon,
                "test_kl_loss": loss_generator_kl,
                "test_adversarial_loss": loss_generator_adversarial,
                "test_loss": loss_generator,
            }
        )

    def on_test_epoch_end(self):
        epoch_mean = np.mean(self.test_step_outputs)
        self.log("test_epoch_loss", epoch_mean, on_step=False, on_epoch=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        optimizer_g = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer_g, optimizer_d
