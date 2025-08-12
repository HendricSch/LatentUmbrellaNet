import torch
import lightning as pl
import torch.nn as nn

from models.blocks.encoder import Encoder
from models.blocks.decoder import Decoder

from models.blocks.distributions import DiagonalGaussianDistribution
from models.losses import RecKLDiscriminatorLoss

from metrics.metrics import WeightedRMSE


class Autoencoder(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()

        ### Load the configuration ###

        self.config = config["config"]

        # data
        self.x = self.config["data"]["x"]
        self.y = self.config["data"]["y"]
        self.in_channels = self.config["data"]["in_channels"]
        self.out_channels = self.config["data"]["out_channels"]

        # model
        self.z_channels = self.config["model"]["z_channels"]
        self.embed_dim = self.config["model"]["embed_dim"]
        self.channels = self.config["model"]["channels"]
        self.channel_mult = self.config["model"]["channel_mult"]
        self.num_res_blocks = self.config["model"]["num_res_blocks"]
        self.attention = self.config["model"]["attention"]

        # loss
        self.reconstruction_loss_fn = self.config["loss"]["reconstruction_loss"]
        self.kl_weight = self.config["loss"]["kl_weight"]
        self.weighted_rmse = WeightedRMSE(num_latitudes=720)

        # training
        self.learning_rate = self.config["training"]["learning_rate"]
        self.epochs = self.config["training"]["epochs"]
        self.batch_size = self.config["training"]["batch_size"]
        self.lr_scheduler = self.config["training"]["lr_scheduler"]

        # lightning
        self.save_hyperparameters()
        self.lr = self.learning_rate
        self.example_input_array = torch.zeros(
            self.batch_size, self.in_channels, self.x, self.y
        )
        self.automatic_optimization = False

        ### Initialize the model ###

        # Encoder and decoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Quantization layers
        self.quant_conv = torch.nn.Conv2d(2 * self.z_channels, 2 * self.embed_dim, 1)

        # Post-quantization layers
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.z_channels, 1)

        # Loss and discriminator
        self.loss_model = RecKLDiscriminatorLoss(config, device="cuda")

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)

        posterior = DiagonalGaussianDistribution(moments)

        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        rec = self.decoder(z)

        return rec

    def forward(
        self, input, sample_posterior=True
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(input)

        if sample_posterior:
            z = posterior.sample()

        else:
            z = posterior.mode()

        dec = self.decode(z)

        return dec, posterior

    def training_step(self, batch: torch.Tensor, batch_idx):
        opt_ae, opt_disc = self.optimizers()

        inputs = batch
        reconstructions, posteriors = self.forward(inputs)

        ### Train encoder + decoder ###
        ae_loss, ae_log = self.loss_model.forward(
            inputs=inputs,
            reconstructions=reconstructions,
            posteriors=posteriors,
            global_step=self.global_step,
            optimizer="Generator",
            last_layer=self.get_last_layer(),
        )

        opt_ae.zero_grad()
        self.manual_backward(ae_loss)
        opt_ae.step()

        ### Train discriminator ###
        disc_loss, disc_log = self.loss_model.forward(
            inputs=inputs,
            reconstructions=reconstructions,
            posteriors=posteriors,
            global_step=self.global_step,
            optimizer="Discriminator",
            last_layer=self.get_last_layer(),
        )

        opt_disc.zero_grad()
        self.manual_backward(disc_loss)
        opt_disc.step()

        ### Log the losses ###
        self.log("vae_loss", ae_loss, prog_bar=True)

        self.log_dict(ae_log)
        self.log_dict(disc_log)

    def validation_step(self, batch: torch.Tensor, batch_idx):
        inputs = batch
        reconstructions, posteriors = self.forward(inputs)

        loss, ae_log = self.loss_model.forward(
            inputs=inputs,
            reconstructions=reconstructions,
            posteriors=posteriors,
            global_step=self.global_step,
            optimizer="Generator",
            last_layer=self.get_last_layer(),
        )

        # rmse for z500

        z500_truth = inputs[:, 50, :, :]
        z500_pred = reconstructions[:, 50, :, :]

        z500_mean, z500_std = 53921.2137, 3091.97738

        z_500_truth = (z500_truth * z500_std) + z500_mean
        z_500_pred = (z500_pred * z500_std) + z500_mean

        z_500_truth = z_500_truth.detach().cpu().numpy()
        z_500_pred = z_500_pred.detach().cpu().numpy()

        rmse = self.weighted_rmse(z_500_truth, z_500_pred)[0]
        rmse = float(rmse)

        self.log_dict(
            {"val_vae_loss": ae_log["rec_loss"], "val_rmse_z500": rmse}, prog_bar=True
        )

    def configure_optimizers(self):
        lr = self.learning_rate

        opt_ae = torch.optim.AdamW(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )

        opt_disc = torch.optim.AdamW(
            self.loss_model.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )

        return [opt_ae, opt_disc], []


class VQAutoencoder(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()

        ### Load the configuration ###

        self.config = config["config"]

        # data
        self.x = self.config["data"]["x"]
        self.y = self.config["data"]["y"]
        self.in_channels = self.config["data"]["in_channels"]
        self.out_channels = self.config["data"]["out_channels"]

        # model
        self.z_channels = self.config["model"]["z_channels"]
        self.embed_dim = self.config["model"]["embed_dim"]
        self.channels = self.config["model"]["channels"]
        self.channel_mult = self.config["model"]["channel_mult"]
        self.num_res_blocks = self.config["model"]["num_res_blocks"]
        self.attention = self.config["model"]["attention"]

        # loss
        self.reconstruction_loss_fn = self.config["loss"]["reconstruction_loss"]
        self.commitment_beta = self.config["model"]["commitment_beta"]
        self.codebook_size = self.config["model"]["codebook_size"]

        # training
        self.learning_rate = self.config["training"]["learning_rate"]
        self.epochs = self.config["training"]["epochs"]
        self.batch_size = self.config["training"]["batch_size"]
        self.lr_scheduler = self.config["training"]["lr_scheduler"]

        # lightning
        self.save_hyperparameters()
        self.lr = self.learning_rate
        self.example_input_array = torch.zeros(
            self.batch_size, self.in_channels, self.x, self.y
        )

        ### Initialize the model ###

        # Encoder and decoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.embedding = nn.Embedding(self.codebook_size, self.embed_dim)

        # Quantization layers
        self.quant_conv = torch.nn.Conv2d(2 * self.z_channels, self.embed_dim, 1)

        # Post-quantization layers
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.z_channels, 1)

    def quantize(self, x):
        # reference: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py

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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        out = self.encoder(x)

        out, quant_losses, _ = self.quantize(out)

        return out, quant_losses

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        rec = self.decoder(z)

        return rec

    def forward(self, input) -> tuple[torch.Tensor, torch.Tensor, dict]:
        z, quant_losses = self.encode(input)
        out = self.decode(z)
        return out, z, quant_losses

    def training_step(self, batch: torch.Tensor, batch_idx):
        inputs = batch
        reconstructions, latent, quant_losses = self.forward(inputs)

        ae_loss = (
            self.reconstruction_loss_fn(reconstructions, inputs)
            + self.commitment_beta * quant_losses["commitment_loss"]
            + quant_losses["codebook_loss"]
        )

        ### Log the losses ###
        self.log("vae_loss", ae_loss, prog_bar=True)

        self.log_dict(quant_losses)

    def validation_step(self, batch: torch.Tensor, batch_idx):
        inputs = batch
        reconstructions, _, _ = self.forward(inputs)

        rec_loss = self.reconstruction_loss_fn(reconstructions, inputs)

        # rmse for z500

        z500_truth = inputs[:, 50, :, :]
        z500_pred = reconstructions[:, 50, :, :]

        z500_mean, z500_std = 53921.2137, 3091.97738

        z_500_truth = (z500_truth * z500_std) + z500_mean
        z_500_pred = (z500_pred * z500_std) + z500_mean

        z_500_truth = z_500_truth.detach().cpu().numpy()
        z_500_pred = z_500_pred.detach().cpu().numpy()

        rmse = self.weighted_rmse(z_500_truth, z_500_pred)[0]
        rmse = float(rmse)

        self.log_dict({"val_vae_loss": rec_loss, "val_rmse_z500": rmse}, prog_bar=True)

    def configure_optimizers(self):
        lr = self.learning_rate

        opt = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(0.5, 0.9),
        )

        return [opt], []
