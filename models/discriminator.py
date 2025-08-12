# reference: https://arxiv.org/abs/1803.07422

import torch.nn as nn
import functools


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels: int = 64):
        super(PatchGANDiscriminator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,  # 69
                out_channels=channels * 2,  # 128
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 2,  # 128
                out_channels=channels * 2,  # 128
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 2,  # 128
                out_channels=channels * 4,  # 256
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 4,  # 256
                out_channels=channels * 8,  # 512
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 8,  # 512
                out_channels=channels * 8,  # 512
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 8,  # 512
                out_channels=out_channels,  # 69
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.out(x)
        return x
