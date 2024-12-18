import torch
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


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers=2,
        num_heads=4,
        norm_channels=32,
    ):
        super(MidBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.norm_channels = norm_channels

        self.resnet_blocks = nn.ModuleList(
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
                for i in range(num_layers + 1)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x):
        out = x

        # First resnet block
        resnet_input = out
        out = self.resnet_blocks[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            # Attention
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            # Resnet block
            resnet_input = out
            out = self.resnet_blocks[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


def main():
    x = torch.randn(32, 32, 128, 64)
    down_block_1 = DownBlock(32, 64)
    down_block_2 = DownBlock(64, 128)
    down_block_3 = DownBlock(128, 256)
    mid_block_1 = MidBlock(256, 512)
    mid_block_2 = MidBlock(512, 256)
    up_block_1 = UpBlock(256, 128)
    up_block_2 = UpBlock(128, 64)
    up_block_3 = UpBlock(64, 32)

    out = down_block_1(x)
    out = down_block_2(out)
    out = down_block_3(out)
    out = mid_block_1(out)
    out = mid_block_2(out)
    out = up_block_1(out)
    out = up_block_2(out)
    out = up_block_3(out)
    print(out.shape)


if __name__ == "__main__":
    main()
