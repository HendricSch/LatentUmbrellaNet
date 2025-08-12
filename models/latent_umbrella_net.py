from models.autoencoder import Autoencoder
from models.predictionnet import PredictionModel, AFNOPredictionModel
from models.blocks.distributions import DiagonalGaussianDistribution

import torch.nn as nn
import numpy as np
import torch
from torchvision.transforms import Normalize
import yaml


class LatentUmbrellaNet(nn.Module):
    def __init__(
        self,
        *,
        vae_ckpt_path: str,
        vae_config_path: str,
        prediction_net_type: str,
        prediction_net_ckpt_path: str,
        device: str,
    ):
        super(LatentUmbrellaNet, self).__init__()

        self.device = device

        # Load the VAE model from the checkpoint and freeze its parameters
        with open(vae_config_path, "r") as f:
            vae_config = yaml.load(f, Loader=yaml.FullLoader)

        self.vae: Autoencoder = Autoencoder.load_from_checkpoint(
            vae_ckpt_path, config=vae_config, strict=False
        ).to(device)
        self.vae.freeze()

        if prediction_net_type == "afno":
            # Load the prediction model from the checkpoint and freeze its parameters
            self.prediction_net: AFNOPredictionModel = (
                AFNOPredictionModel.load_from_checkpoint(
                    prediction_net_ckpt_path, strict=False
                ).to(device)
            )
            self.prediction_net.freeze()

        elif prediction_net_type == "unet":
            # Load the prediction model from the checkpoint and freeze its parameters
            self.prediction_net: PredictionModel = PredictionModel.load_from_checkpoint(
                prediction_net_ckpt_path, strict=False
            ).to(device)
            self.prediction_net.freeze()

    @staticmethod
    def normalize_input(x: torch.Tensor) -> torch.Tensor:
        mean = np.array(
            [
                2.76707305e02,
                -1.02550652e-01,
                -8.24716593e-02,
                1.01068682e05,
                2.13901834e02,
                2.09669021e02,
                2.14224057e02,
                2.18012186e02,
                2.22117960e02,
                2.27618180e02,
                2.40553091e02,
                2.51450718e02,
                2.59819244e02,
                2.66263193e02,
                2.73431732e02,
                2.76170563e02,
                2.79528167e02,
                3.87254235e00,
                9.39696721e00,
                1.39809760e01,
                1.49588660e01,
                1.42096134e01,
                1.26746424e01,
                9.40749201e00,
                6.76743938e00,
                4.85057830e00,
                3.21840022e00,
                1.19613039e00,
                3.40955153e-01,
                -2.00982027e-01,
                1.34509794e-01,
                1.86537438e-02,
                1.77366811e-01,
                2.60285472e-01,
                1.08158604e-01,
                2.13348037e-02,
                -5.33151006e-02,
                -1.12940699e-02,
                1.37653121e-02,
                1.64470187e-02,
                -5.36961093e-03,
                -1.42718665e-02,
                -8.16306830e-02,
                1.99295885e05,
                1.57330177e05,
                1.32683094e05,
                1.14840669e05,
                1.00754974e05,
                8.89962866e04,
                6.96855338e04,
                5.39212137e04,
                4.05297225e04,
                2.88684465e04,
                1.37619912e04,
                7.06023469e03,
                8.15529195e02,
                2.87899168e-06,
                2.44946302e-06,
                4.41716612e-06,
                1.54408574e-05,
                4.63313069e-05,
                1.05735979e-04,
                3.32204274e-04,
                7.38973747e-04,
                1.37365580e-03,
                2.20929030e-03,
                4.23163159e-03,
                5.59333540e-03,
                6.48287372e-03,
            ]
        )

        std = np.array(
            [
                2.09942404e01,
                5.25000636e00,
                4.54455487e00,
                1.30960034e03,
                8.97812032e00,
                1.32565826e01,
                8.31339312e00,
                5.15994231e00,
                6.88576031e00,
                9.93203450e00,
                1.24352490e01,
                1.29195538e01,
                1.30728671e01,
                1.40098769e01,
                1.47487644e01,
                1.53852921e01,
                1.71116930e01,
                1.00916061e01,
                1.18567912e01,
                1.51044572e01,
                1.70482496e01,
                1.72106285e01,
                1.64754925e01,
                1.39160706e01,
                1.17258202e01,
                1.00555255e01,
                8.94536813e00,
                7.80402390e00,
                7.49754381e00,
                5.91365735e00,
                7.13226032e00,
                7.68995984e00,
                9.47791003e00,
                1.15089522e01,
                1.27980862e01,
                1.27539256e01,
                1.08107437e01,
                8.95480061e00,
                7.69034815e00,
                6.91974370e00,
                6.33759832e00,
                6.47175201e00,
                5.22074238e00,
                2.97759049e03,
                3.99086247e03,
                4.97500846e03,
                5.28610563e03,
                5.15772933e03,
                4.77762842e03,
                3.87501782e03,
                3.09197738e03,
                2.45088338e03,
                1.91940426e03,
                1.30757654e03,
                1.10889327e03,
                1.01593943e03,
                1.87661911e-07,
                4.75091686e-07,
                2.64407777e-06,
                1.75901199e-05,
                6.03882715e-05,
                1.42577167e-04,
                4.54680063e-04,
                9.75985021e-04,
                1.64347251e-03,
                2.37802664e-03,
                3.98016829e-03,
                4.98595989e-03,
                5.80280740e-03,
            ]
        )

        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)

        normalizer = Normalize(mean=mean, std=std)

        return normalizer(x).float()

    @staticmethod
    def denormalize_input(x: torch.Tensor) -> torch.Tensor:
        mean = np.array(
            [
                2.76707305e02,
                -1.02550652e-01,
                -8.24716593e-02,
                1.01068682e05,
                2.13901834e02,
                2.09669021e02,
                2.14224057e02,
                2.18012186e02,
                2.22117960e02,
                2.27618180e02,
                2.40553091e02,
                2.51450718e02,
                2.59819244e02,
                2.66263193e02,
                2.73431732e02,
                2.76170563e02,
                2.79528167e02,
                3.87254235e00,
                9.39696721e00,
                1.39809760e01,
                1.49588660e01,
                1.42096134e01,
                1.26746424e01,
                9.40749201e00,
                6.76743938e00,
                4.85057830e00,
                3.21840022e00,
                1.19613039e00,
                3.40955153e-01,
                -2.00982027e-01,
                1.34509794e-01,
                1.86537438e-02,
                1.77366811e-01,
                2.60285472e-01,
                1.08158604e-01,
                2.13348037e-02,
                -5.33151006e-02,
                -1.12940699e-02,
                1.37653121e-02,
                1.64470187e-02,
                -5.36961093e-03,
                -1.42718665e-02,
                -8.16306830e-02,
                1.99295885e05,
                1.57330177e05,
                1.32683094e05,
                1.14840669e05,
                1.00754974e05,
                8.89962866e04,
                6.96855338e04,
                5.39212137e04,
                4.05297225e04,
                2.88684465e04,
                1.37619912e04,
                7.06023469e03,
                8.15529195e02,
                2.87899168e-06,
                2.44946302e-06,
                4.41716612e-06,
                1.54408574e-05,
                4.63313069e-05,
                1.05735979e-04,
                3.32204274e-04,
                7.38973747e-04,
                1.37365580e-03,
                2.20929030e-03,
                4.23163159e-03,
                5.59333540e-03,
                6.48287372e-03,
            ]
        )

        std = np.array(
            [
                2.09942404e01,
                5.25000636e00,
                4.54455487e00,
                1.30960034e03,
                8.97812032e00,
                1.32565826e01,
                8.31339312e00,
                5.15994231e00,
                6.88576031e00,
                9.93203450e00,
                1.24352490e01,
                1.29195538e01,
                1.30728671e01,
                1.40098769e01,
                1.47487644e01,
                1.53852921e01,
                1.71116930e01,
                1.00916061e01,
                1.18567912e01,
                1.51044572e01,
                1.70482496e01,
                1.72106285e01,
                1.64754925e01,
                1.39160706e01,
                1.17258202e01,
                1.00555255e01,
                8.94536813e00,
                7.80402390e00,
                7.49754381e00,
                5.91365735e00,
                7.13226032e00,
                7.68995984e00,
                9.47791003e00,
                1.15089522e01,
                1.27980862e01,
                1.27539256e01,
                1.08107437e01,
                8.95480061e00,
                7.69034815e00,
                6.91974370e00,
                6.33759832e00,
                6.47175201e00,
                5.22074238e00,
                2.97759049e03,
                3.99086247e03,
                4.97500846e03,
                5.28610563e03,
                5.15772933e03,
                4.77762842e03,
                3.87501782e03,
                3.09197738e03,
                2.45088338e03,
                1.91940426e03,
                1.30757654e03,
                1.10889327e03,
                1.01593943e03,
                1.87661911e-07,
                4.75091686e-07,
                2.64407777e-06,
                1.75901199e-05,
                6.03882715e-05,
                1.42577167e-04,
                4.54680063e-04,
                9.75985021e-04,
                1.64347251e-03,
                2.37802664e-03,
                3.98016829e-03,
                4.98595989e-03,
                5.80280740e-03,
            ]
        )

        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)

        x_denorm = x * std[None, :, None, None] + mean[None, :, None, None]

        return x_denorm

    @torch.no_grad()
    def forward(
        self, x_1: torch.Tensor, x_2: torch.Tensor, forecast_steps: int = 1
    ) -> torch.Tensor:
        if x_1.shape == (1, 69, 1440, 721):
            x_1 = x_1[:, :, :, :-1]
            x_2 = x_2[:, :, :, :-1]

        x_1 = x_1.to(self.device)
        x_2 = x_2.to(self.device)

        # Normalize the input data
        x_1 = self.normalize_input(x_1)
        x_2 = self.normalize_input(x_2)

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            # Pass the inputs through the VAE model to get the moments (mean and logvar)
            moments_x_1 = self.vae.encoder.forward(x_1)
            moments_x_2 = self.vae.encoder.forward(x_2)

            moments_x_1 = self.vae.quant_conv.forward(moments_x_1)
            moments_x_2 = self.vae.quant_conv.forward(moments_x_2)

            moments = torch.cat([moments_x_1, moments_x_2], dim=1)

            # Pass the moments through the prediction network to get the next prediction
            for i in range(forecast_steps):
                pred = self.prediction_net.forward(moments)
                _, prev_moments_2 = moments.chunk(2, dim=1)
                moments = torch.cat([prev_moments_2, pred], dim=1)

            z = DiagonalGaussianDistribution(pred).sample()

            # Decode the latent variable to get the final output
            output = self.vae.decode(z)

        # Denormalize the output data
        output = output.to(torch.float32).detach().cpu()
        output = self.denormalize_input(output)

        return output

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Nur VAE-Durchlauf: Encodiert und dekodiert Eingaben ohne Vorhersageschritt."""
        if x.shape == (1, 69, 1440, 721):
            x = x[:, :, :, :-1]

        x = x.to(self.device)

        # Normalize the input data
        x = self.normalize_input(x)

        # Pass the input through the VAE model
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            dec, _ = self.vae.forward(x)

        # Denormalize the output data
        output = dec.to(torch.float32).detach().cpu()
        output = self.denormalize_input(output)

        return output


if __name__ == "__main__":
    x_1 = torch.randn(1, 69, 1440, 721)
    x_2 = torch.randn(1, 69, 1440, 721)

    lun = LatentUmbrellaNet(
        vae_ckpt_path="checkpoints/vae-kl-f8-rmse-disc-2-step=5000-z500=93.ckpt",
        vae_config_path="configs/autoencoder/kl-f8-disc.yaml",
        prediction_net_ckpt_path="lightning_logs/version_39/checkpoints/epoch=9-step=7200.ckpt",
        device="cuda",
    )

    output = lun(x_1, x_2)
    print(output.shape)  # Should be (1, 69, 1440, 721)
