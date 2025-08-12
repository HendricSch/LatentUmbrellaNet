import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_layers=2, norm_channels=32
    ):
        super(DownBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.norm_channels = norm_channels

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        num_groups=self.norm_channels,
                        num_channels=self.in_channels if i == 0 else self.out_channels,
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels=self.in_channels if i == 0 else self.out_channels,
                        out_channels=self.out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GroupNorm(
                        num_groups=self.norm_channels, num_channels=self.out_channels
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels=self.out_channels,
                        out_channels=self.out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(self.num_layers)
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

        self.downsample_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        out = x

        for i in range(self.num_layers):
            resnet_input = out
            out = self.layers[i](out)
            out += self.residual_input_conv[i](resnet_input)

        return self.downsample_conv(out)
