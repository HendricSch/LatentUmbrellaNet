import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding='same',
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding='same',
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)

        return x


class ResNet(nn.Module):
    def __init__(self, in_channels: int):
        super(ResNet, self).__init__()

        self.init_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=(3, 3),
                padding='same',
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.layer_1 = nn.Sequential(
            ResidualBlock(in_channels=16, out_channels=16),
            ResidualBlock(in_channels=16, out_channels=16),
            ResidualBlock(in_channels=16, out_channels=16)
        )

        self.layer_2 = nn.Sequential(
            ResidualBlock(in_channels=16, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32)
        )

        self.layer_3 = nn.Sequential(
            ResidualBlock(in_channels=32, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64)
        )

        self.layer_4 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32)
        )

        self.layer_5 = nn.Sequential(
            ResidualBlock(in_channels=32, out_channels=16),
            ResidualBlock(in_channels=16, out_channels=16),
            ResidualBlock(in_channels=16, out_channels=16)
        )

        self.final_layer = nn.Conv2d(
            in_channels=16,
            out_channels=in_channels,
            kernel_size=(3, 3),
            padding='same'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.final_layer(x)

        return x
