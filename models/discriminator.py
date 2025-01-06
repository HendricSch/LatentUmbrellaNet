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
                    self.activation if i != len(
                        self.dims) - 2 else nn.Identity(),
                )
                for i in range(len(self.dims) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        for layer in self.layers:
            out = layer(out)

        return out


class PatchGanDiscriminator(nn.Module):

    def __init__(self, in_channels: int):
        super(PatchGanDiscriminator, self).__init__()

        self.c64 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(
                4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2)
        )

        self.c128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        # self.c256 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2)
        # )

        # self.c512 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.2)
        # )

        self.second_last = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(4, 4),
                      stride=(1, 1), padding="same"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.output = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=(4, 4),
                      stride=(1, 1), padding="same"),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.c64(x)
        x = self.c128(x)
        # x = self.c256(x)
        # x = self.c512(x)
        x = self.second_last(x)
        x = self.output(x)

        return x


def main():
    model = Discriminator()
    x = torch.rand(32, 1, 128, 64)
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    main()
