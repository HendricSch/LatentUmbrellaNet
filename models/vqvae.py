import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
import torchvision

from models.blocks import DownBlock, MidBlock, UpBlock
from models.discriminator import Discriminator


class VQVAE(pl.LightningModule):
    def __init__(self, in_channels: int) -> None:
        super(VQVAE, self).__init__()

        # Hyperparameters Lightining
        self.lr = 0.00001
        self.example_input_array = torch.rand(1, 1, 128, 64)
        self.automatic_optimization = False

        self.step_count = 0

        # Hyperparameters VQVAE
        self.down_channels = [64, 128, 256, 256]
        self.mid_channels = [256, 256]
        self.up_channels = list(reversed(self.down_channels))
        self.in_channels = in_channels
        self.norm_groups = 32
        self.latent_dim = 3
        self.codebook_size = 8192
        self.codebook_weight = 1.0
        self.commitment_weight = 0.2

        # Discriminator
        self.discriminator = Discriminator(in_channels=1)
        self.discriminator_start_step = 2000
        self.disc_weight = 0.5

        ### Encoder ###
        self.encoder_conv_in = nn.Conv2d(
            self.in_channels, self.down_channels[0], kernel_size=3, padding=1
        )

        self.encoder_down_blocks = nn.ModuleList(
            [
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    num_layers=2,
                    norm_channels=self.norm_groups,
                )
                for i in range(len(self.down_channels) - 1)
            ]
        )

        self.encoder_mid_blocks = nn.ModuleList(
            [
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    num_layers=2,
                    norm_channels=self.norm_groups,
                )
                for i in range(len(self.mid_channels) - 1)
            ]
        )

        self.encoder_norm_out = nn.GroupNorm(self.norm_groups, self.down_channels[-1])

        self.encoder_conv_out = nn.Conv2d(
            self.down_channels[-1], self.latent_dim, kernel_size=3, padding=1
        )

        self.pre_quant_conv = nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=1)

        self.embedding = nn.Embedding(self.codebook_size, self.latent_dim)

        ### Decoder ###
        self.post_quant_conv = nn.Conv2d(
            self.latent_dim, self.latent_dim, kernel_size=1
        )

        self.decoder_conv_in = nn.Conv2d(
            self.latent_dim, self.up_channels[0], kernel_size=3, padding=1
        )

        # MidBlock + UpBlocks

        self.decoder_mid_blocks = nn.ModuleList(
            [
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    num_layers=2,
                    norm_channels=self.norm_groups,
                )
                for i in range(len(self.mid_channels) - 1)
            ]
        )

        self.decoder_up_blocks = nn.ModuleList(
            [
                UpBlock(
                    self.up_channels[i],
                    self.up_channels[i + 1],
                    num_layers=2,
                    norm_channels=self.norm_groups,
                )
                for i in range(len(self.up_channels) - 1)
            ]
        )

        self.decoder_norm_out = nn.GroupNorm(self.norm_groups, self.up_channels[-1])

        self.decoder_conv_out = nn.Conv2d(
            self.up_channels[-1], self.in_channels, kernel_size=3, padding=1
        )

    def quantize(self, x):
        B, C, H, W = x.shape

        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)

        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))

        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)

        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(
            self.embedding.weight, 0, min_encoding_indices.view(-1)
        )

        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            "codebook_loss": codebook_loss,
            "commitment_loss": commmitment_loss,
        }
        # Straight through estimation
        quant_out = x + (quant_out - x).detach()

        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape(
            (-1, quant_out.size(-2), quant_out.size(-1))
        )
        return quant_out, quantize_losses, min_encoding_indices

    def encode(self, x):
        out = self.encoder_conv_in(x)

        for block in self.encoder_down_blocks:
            out = block(out)

        for block in self.encoder_mid_blocks:
            out = block(out)

        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)

        out, quant_losses, _ = self.quantize(out)

        return out, quant_losses

    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)

        for block in self.decoder_mid_blocks:
            out = block(out)

        for block in self.decoder_up_blocks:
            out = block(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)

        return out

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, dict]:
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        return out, z, quant_losses

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

    def calc_losses_generator(self, sample, prediction) -> dict:
        # Reconstruction loss
        recon_loss = F.mse_loss(prediction, sample)

        # Adversarial loss
        disc_fake_pred = self.discriminator(prediction)
        adversarial_loss = F.mse_loss(
            disc_fake_pred,
            torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device),
        )
        adversarial_loss = self.disc_weight * adversarial_loss

        loss_dict = {
            "recon_loss": recon_loss,
            "adversarial_loss": adversarial_loss,
        }

        return loss_dict

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()

        sample = batch
        prediction, _, quant_losses = self.forward(sample)

        # Calculate losses for generator
        loss_generator_dict = self.calc_losses_generator(sample, prediction)
        loss_generator_recon = loss_generator_dict["recon_loss"]
        loss_generator_adversarial = loss_generator_dict["adversarial_loss"]
        loss_generator_codebook = quant_losses["codebook_loss"] * self.codebook_weight
        loss_generator_commitment = (
            quant_losses["commitment_loss"] * self.commitment_weight
        )

        if self.step_count > self.discriminator_start_step:
            loss_generator = (
                loss_generator_recon
                + loss_generator_adversarial
                + loss_generator_codebook
                + loss_generator_commitment
            )
        else:
            loss_generator = (
                loss_generator_recon
                + loss_generator_codebook
                + loss_generator_commitment
            )

        # Train generator
        optimizer_g.zero_grad()
        self.manual_backward(loss_generator)
        optimizer_g.step()

        # Calculate losses for discriminator
        loss_discriminator = self.calc_losses_discriminator(sample, prediction)

        if self.step_count > self.discriminator_start_step:
            # Train discriminator
            optimizer_d.zero_grad()
            self.manual_backward(loss_discriminator)
            optimizer_d.step()

        self.log_dict(
            {
                "train_recon_loss": loss_generator_recon,
                "train_adversarial_loss": loss_generator_adversarial,
                "train_generator_loss": loss_generator,
                "train_disc_loss": loss_discriminator,
                "train_codebook_loss": loss_generator_codebook,
                "train_commitment_loss": loss_generator_commitment,
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

    def validation_step(self, batch, batch_idx):
        sample = batch
        prediction, _, quant_losses = self.forward(sample)

        # Calculate losses for generator
        loss_generator_dict = self.calc_losses_generator(sample, prediction)
        loss_generator_recon = loss_generator_dict["recon_loss"]
        loss_generator_adversarial = loss_generator_dict["adversarial_loss"]
        loss_generator_codebook = quant_losses["codebook_loss"] * self.codebook_weight
        loss_generator_commitment = (
            quant_losses["commitment_loss"] * self.commitment_weight
        )

        if self.step_count > self.discriminator_start_step:
            loss_generator = (
                loss_generator_recon
                + loss_generator_adversarial
                + loss_generator_codebook
                + loss_generator_commitment
            )
        else:
            loss_generator = (
                loss_generator_recon
                + loss_generator_codebook
                + loss_generator_commitment
            )

        # Calculate losses for discriminator
        loss_discriminator = self.calc_losses_discriminator(sample, prediction)

        self.log_dict(
            {
                "val_recon_loss": loss_generator_recon,
                "val_adversarial_loss": loss_generator_adversarial,
                "val_generator_loss": loss_generator,
                "val_disc_loss": loss_discriminator,
                "val_codebook_loss": loss_generator_codebook,
                "val_commitment_loss": loss_generator_commitment,
            }
        )

    def test_step(self, batch, batch_idx):
        sample = batch
        prediction, _, quant_losses = self.forward(sample)

        # Calculate losses for generator
        loss_generator_dict = self.calc_losses_generator(sample, prediction)
        loss_generator_recon = loss_generator_dict["recon_loss"]
        loss_generator_adversarial = loss_generator_dict["adversarial_loss"]
        loss_generator_codebook = quant_losses["codebook_loss"] * self.codebook_weight
        loss_generator_commitment = (
            quant_losses["commitment_loss"] * self.commitment_weight
        )

        if self.step_count > self.discriminator_start_step:
            loss_generator = (
                loss_generator_recon
                + loss_generator_adversarial
                + loss_generator_codebook
                + loss_generator_commitment
            )
        else:
            loss_generator = (
                loss_generator_recon
                + loss_generator_codebook
                + loss_generator_commitment
            )

        # Calculate losses for discriminator
        loss_discriminator = self.calc_losses_discriminator(sample, prediction)

        self.log_dict(
            {
                "test_recon_loss": loss_generator_recon,
                "test_adversarial_loss": loss_generator_adversarial,
                "test_generator_loss": loss_generator,
                "test_disc_loss": loss_discriminator,
                "test_codebook_loss": loss_generator_codebook,
                "test_commitment_loss": loss_generator_commitment,
            }
        )

    def configure_optimizers(self):
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        optimizer_g = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer_g, optimizer_d
