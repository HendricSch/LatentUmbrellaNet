import torch.nn as nn


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers=2,
        num_heads=4,
        norm_channels=32,
        attention=True
    ):
        super(MidBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.norm_channels = norm_channels
        self.attention = attention

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

        if self.attention:

            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )

            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        out_channels, num_heads, batch_first=True)
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
            if self.attention:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(
                    batch_size, channels, h, w)
                out = out + out_attn

            # Resnet block
            resnet_input = out
            out = self.resnet_blocks[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out
