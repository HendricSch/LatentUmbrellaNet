import torch.nn as nn


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_layers=2, norm_channels=32
    ):
        super(UpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.norm_channels = norm_channels

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        norm_channels, in_channels if i == 0 else out_channels
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for i in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.in_channels if i == 0 else self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                )
                for i in range(self.num_layers)
            ]
        )

        self.upsample_conv = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        x = self.upsample_conv(x)

        out = x

        for i in range(self.num_layers):
            resnet_input = out
            out = self.layers[i](out)
            out += self.residual_input_conv[i](resnet_input)

        return out
