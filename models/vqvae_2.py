import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import OneCycleLR

from models.blocks import DownBlock, MidBlock, UpBlock


class VQVAE2(pl.LightningModule):
    def __init__(
        self, in_channels: int, lr: float, latent_dim: int, codebook_size: int
    ) -> None:
        super(VQVAE2, self).__init__()

        self.save_hyperparameters()

        # Hyperparameters Lightining
        self.lr = lr
        self.example_input_array = torch.rand(1, 5, 128, 64)

        # Hyperparameters VQVAE
        self.down_channels = [32, 64, 128, 128]
        self.mid_channels = [128, 128]
        self.up_channels = list(reversed(self.down_channels))
        self.in_channels = in_channels
        self.norm_groups = 32
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.codebook_weight = 1.0
        self.commitment_weight = 0.2

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

        self.encoder_norm_out = nn.GroupNorm(
            self.norm_groups, self.down_channels[-1])

        self.encoder_conv_out = nn.Conv2d(
            self.down_channels[-1], self.latent_dim, kernel_size=3, padding=1
        )

        self.pre_quant_conv = nn.Conv2d(
            self.latent_dim, self.latent_dim, kernel_size=1)

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

        self.decoder_norm_out = nn.GroupNorm(
            self.norm_groups, self.up_channels[-1])

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
        dist = torch.cdist(
            x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
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

    def training_step(self, batch, batch_idx):
        sample = batch
        prediction, _, quant_losses = self.forward(sample)

        loss_recon = F.mse_loss(prediction, sample)
        loss_codebook = quant_losses["codebook_loss"] * self.codebook_weight
        loss_commitment = quant_losses["commitment_loss"] * \
            self.commitment_weight

        loss = loss_recon + loss_codebook + loss_commitment

        self.log_dict(
            {
                "train_recon_loss": loss_recon,
                "train_codebook_loss": loss_codebook,
                "train_commitment_loss": loss_commitment,
                "train_loss": loss,
            }
        )

        # log an image of the reconstruction every 100 steps to tensorboard
        if batch_idx % 100 == 0:
            sample_plot = sample[0:3, 0, :, :].unsqueeze(1)
            prediction_plot = prediction[0:3, 0, :, :].unsqueeze(1)
            self.logger.experiment.add_image(
                "c500",
                torchvision.utils.make_grid(
                    torch.cat([sample_plot, prediction_plot], dim=0)
                ),
                self.current_epoch,
            )

            sample_plot = sample[0:3, 3, :, :].unsqueeze(1)
            prediction_plot = prediction[0:3, 3, :, :].unsqueeze(1)
            self.logger.experiment.add_image(
                "t2m",
                torchvision.utils.make_grid(
                    torch.cat([sample_plot, prediction_plot], dim=0)
                ),
                self.current_epoch,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        sample = batch
        prediction, _, quant_losses = self.forward(sample)

        loss_recon = F.mse_loss(prediction, sample)
        loss_codebook = quant_losses["codebook_loss"] * self.codebook_weight
        loss_commitment = quant_losses["commitment_loss"] * \
            self.commitment_weight

        loss = loss_recon + loss_codebook + loss_commitment

        self.log_dict(
            {
                "val_recon_loss": loss_recon,
                "val_codebook_loss": loss_codebook,
                "val_commitment_loss": loss_commitment,
                "val_loss": loss,
            }
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=2301,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
