import torch
import torch.nn as nn

from models.blocks.midblock import MidBlock
from models.blocks.upblock import UpBlock


class Decoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        ### Load the configuration ###

        self.config = config["config"]

        # data
        self.in_channels = self.config["data"]["in_channels"]

        # model
        self.z_channels = self.config["model"]["z_channels"]
        self.channels = self.config["model"]["channels"]
        self.channel_mult = self.config["model"]["channel_mult"]
        self.num_res_blocks = self.config["model"]["num_res_blocks"]
        self.attention = self.config["model"]["attention"]
        self.num_heads = self.config["model"]["num_heads"]
        self.norm_groups = self.config["model"]["norm_groups"]

        self.up_channels = list(
            reversed([self.channels * mult for mult in self.channel_mult])
        )

        ### Initialize the model ###

        self.conv_in = nn.Conv2d(
            in_channels=self.z_channels,
            out_channels=self.up_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.mid_block_1 = MidBlock(
            in_channels=self.up_channels[0],
            out_channels=self.up_channels[0],
            num_layers=self.num_res_blocks,
            num_heads=self.num_heads,
            norm_channels=self.norm_groups,
            attention=self.attention,
        )

        self.mid_block_2 = MidBlock(
            in_channels=self.up_channels[0],
            out_channels=self.up_channels[0],
            num_layers=self.num_res_blocks,
            num_heads=self.num_heads,
            norm_channels=self.norm_groups,
            attention=self.attention,
        )

        self.up_blocks = nn.ModuleList(
            [
                UpBlock(
                    self.up_channels[i],
                    self.up_channels[i + 1],
                    num_layers=2,
                    norm_channels=self.norm_groups,
                )
                for i in range(len(self.up_channels) - 1)
            ]
        )

        self.norm_out = nn.GroupNorm(self.norm_groups, self.up_channels[-1])

        self.silu = nn.SiLU()

        self.conv_out = nn.Conv2d(
            self.up_channels[-1], self.in_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        out = self.conv_in(out)

        out = self.mid_block_1(out)
        out = self.mid_block_2(out)

        for up_block in self.up_blocks:
            out = up_block(out)

        out = self.norm_out(out)
        out = self.silu(out)
        out = self.conv_out(out)

        return out


def main():
    import yaml

    with open("configs/autoencoder_kl_f8.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    decoder = Decoder(config).to("cuda")

    print(decoder.up_channels)

    for i in range(100):
        x = torch.randn(1, 4, 180, 90, device="cuda")

        z = decoder(x)

        print(z.shape)


if __name__ == "__main__":
    main()
