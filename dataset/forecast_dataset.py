import torch
import numpy as np
import matplotlib.pyplot as plt

from models.vqvae import VQVAE


class ForecastDataset(torch.utils.data.Dataset):

    def __init__(self, path: str, vqvae: VQVAE, channels: int, img_res: tuple[int, int]) -> None:
        super(ForecastDataset, self).__init__()

        self.vqvae = vqvae
        self.device = vqvae.device
        self.path = path

        self.channels = channels
        self.img_res = img_res

        self.mean: np.ndarray = np.array(
            [5.40695777e+04, 2.74162604e+02, 2.41138667e-03, 2.77659178e+02, 6.11351155e+00], dtype=np.float32)
        self.std: np.ndarray = np.array(
            [3.28407787e+03, 1.60311676e+01, 2.39652644e-03, 2.20706169e+01, 3.32614560e+00], dtype=np.float32)

        self.data = self._load_data()
        # self.data = self._normalize()

    def _load_data(self):
        path = self.path
        c = self.channels
        w, h = self.img_res

        data = np.memmap(path, mode="r", dtype=np.float32).reshape(-1, c, w, h)

        return data

    def _normalize(self):
        return (self.data - self.mean[None, :, None, None]) / self.std[None, :, None, None]

    def __len__(self):
        return self.data.shape[0] - 1

    @torch.no_grad()
    def __getitem__(self, idx):
        x1 = self.data[idx]
        x2 = self.data[idx + 1]

        x1 = (x1 - self.mean[:, None, None]) / self.std[:, None, None]
        x2 = (x2 - self.mean[:, None, None]) / self.std[:, None, None]

        x1 = torch.tensor(x1).to(self.device)
        x2 = torch.tensor(x2).to(self.device)

        _, latent_x1, _ = self.vqvae(x1.unsqueeze(0))
        _, latent_x2, _ = self.vqvae(x2.unsqueeze(0))

        latent_x1 = latent_x1.squeeze(0)
        latent_x2 = latent_x2.squeeze(0)

        return latent_x1, latent_x2


def main():

    torch.set_float32_matmul_precision("medium")

    vqvae = VQVAE.load_from_checkpoint(
        "logs/vqvae_5channel/version_46/checkpoints/epoch=201-step=4751.ckpt",
        in_channels=5,
    )

    dataset = ForecastDataset(
        path="train.memmap",
        vqvae=vqvae,
        channels=5,
        img_res=(128, 64)
    )

    x, y = dataset[0]
    print(x.shape, y.shape)

    plt.imshow(x[0].cpu().numpy())
    plt.show()


if __name__ == "__main__":
    main()
