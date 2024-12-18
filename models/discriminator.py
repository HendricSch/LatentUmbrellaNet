import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN discriminator for the training of the VAE model.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: list[int] = [64, 128, 256],
        kernel_sizes: list[int] = [4, 4, 4, 4],
        strides: list[int] = [2, 2, 2, 1],
        paddings: list[int] = [1, 1, 1, 1],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dims = [in_channels] + conv_channels + [1]

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.dims[i],
                        out_channels=self.dims[i + 1],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=paddings[i],
                        bias=False if i != 0 else True,
                    ),
                    nn.BatchNorm2d(self.dims[i + 1])
                    if i != len(self.dims) - 2 and i != 0
                    else nn.Identity(),
                    self.activation if i != len(self.dims) - 2 else nn.Identity(),
                )
                for i in range(len(self.dims) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        for layer in self.layers:
            out = layer(out)

        return out


def main():
    model = Discriminator()
    x = torch.rand(32, 1, 128, 64)
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    main()
