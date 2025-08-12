import torch
import math
import lightning as pl
from models.afnonet import AFNONet


class AFNOPredictionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.afno = AFNONet(
            img_size=(192, 96),
            patch_size=(8, 8),
            in_chans=256,
            out_chans=128,
            num_blocks=8,
        )

        self.pad = PadBlock((3, 3, 6, 6))
        self.unpad = UnPadBlock((3, 3, 6, 6))

        self.learning_rate = 0.0005
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(8, 256, 180, 90)

        self.loss_fn = torch.nn.MSELoss()

        self.val_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gibt die vorhergesagten latenten Darstellungen für den nächsten Zeitschritt zurück."""
        x = self.pad(x)

        x = self.afno.forward(x)

        x = self.unpad(x)

        return x

    def training_step(self, batch: torch.Tensor, batch_idx):
        x_1, x_2, y = batch

        x = torch.cat([x_1, x_2], dim=1)

        prediction = self.forward(x)

        loss = self.loss_fn.forward(prediction, y)

        self.log("train_loss", loss.item(), prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):
        x_1, x_2, y = batch

        x = torch.cat([x_1, x_2], dim=1)

        prediction = self.forward(x)

        loss = self.loss_fn.forward(prediction, y)

        self.val_losses.append(loss.item())

    def on_validation_epoch_end(self):
        losses = torch.tensor(self.val_losses)

        # remove nan values
        losses = losses[~torch.isnan(losses)]

        # remove inf values
        losses = losses[~torch.isinf(losses)]

        avg_loss = torch.mean(losses)

        self.log("val_loss", avg_loss, prog_bar=True)

        self.val_losses = []

    def configure_optimizers(self):
        lr = self.learning_rate

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=optimizer,
        #     total_steps=6000,
        #     max_lr=lr,
        # )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=6000,
            eta_min=0,
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class PredictionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.unet = UNet(
            in_channels=128,
            base_channels=128,
            channels_mult=[1, 1, 2, 2, 4],
            num_resblocks=2,
            attention_layers=[False, False, True, True, True],
            pads=(3, 3, 6, 6),
        )

        self.learning_rate = 5e-6
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(8, 256, 180, 90)

        self.loss_fn = torch.nn.MSELoss()

        self.val_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gibt die vorhergesagten latenten Darstellungen für den nächsten Zeitschritt zurück."""
        return self.unet.forward(x)

    def training_step(self, batch: torch.Tensor, batch_idx):
        x_1, x_2, y = batch

        x = torch.cat([x_1, x_2], dim=1)

        prediction = self.forward(x)

        loss = self.loss_fn.forward(prediction, y)

        self.log("train_loss", loss.item(), prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):
        x_1, x_2, y = batch

        x = torch.cat([x_1, x_2], dim=1)

        prediction = self.forward(x)

        loss = self.loss_fn.forward(prediction, y)

        self.val_losses.append(loss.item())

    def on_validation_epoch_end(self):
        losses = torch.tensor(self.val_losses)

        # remove nan values
        losses = losses[~torch.isnan(losses)]

        # remove inf values
        losses = losses[~torch.isinf(losses)]

        avg_loss = torch.mean(losses)

        self.log("val_loss", avg_loss, prog_bar=True)

        self.val_losses = []

    def configure_optimizers(self):
        lr = self.learning_rate

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=11300,
            eta_min=0,
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class ResBlock(torch.nn.Module):
    """Residualblock mit GroupNorm und SiLU-Aktivierungen."""

    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.GroupNorm(num_groups=32, num_channels=out_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.GroupNorm(num_groups=32, num_channels=out_channels),
        )

        if in_channels == out_channels:
            self.skip_connection = torch.nn.Identity()
        else:
            self.skip_connection = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )

        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers(x)
        h = h + self.skip_connection(x)
        h = self.activation(h)
        return h


class UpSample(torch.nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


class DownSample(torch.nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()

        self.op = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class QKVAttention(torch.nn.Module):
    def __init__(self, n_heads: int):
        super(QKVAttention, self).__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        bs, width, length = qkv.shape

        assert width % (3 * self.n_heads) == 0

        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)

        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )

        return a.reshape(bs, -1, length)


class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels: int, num_heads: int):
        super(AttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels)

        self.qkv = torch.nn.Conv1d(in_channels, 3 * in_channels, kernel_size=1)

        self.attention = QKVAttention(num_heads)

        self.proj_out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape

        x = x.reshape(b, c, -1)

        qkv = self.qkv(self.norm(x))

        h = self.attention(qkv)
        h = self.proj_out(h)

        h = (x + h).reshape(b, c, *spatial)

        return h


class DownBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_resblocks: int = 2,
        use_attention: bool = True,
    ):
        super(DownBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_resblocks = num_resblocks
        self.use_attention = use_attention

        self.resblocks = torch.nn.ModuleList()
        self.resblocks.append(ResBlock(in_channels, out_channels))

        for _ in range(num_resblocks - 1):
            self.resblocks.append(ResBlock(out_channels, out_channels))

        self.attentionblocks = torch.nn.ModuleList()

        for _ in range(num_resblocks):
            if use_attention:
                self.attentionblocks.append(AttentionBlock(out_channels, num_heads=8))
            else:
                self.attentionblocks.append(torch.nn.Identity())

        self.down = DownSample()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        for i in range(self.num_resblocks):
            h = self.resblocks[i](h)
            h = self.attentionblocks[i](h)

        # return after downsampling and skip connection
        return self.down(h), h


class UpBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_resblocks: int = 2,
        use_attention: bool = True,
    ):
        super(UpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_resblocks = num_resblocks
        self.use_attention = use_attention

        self.up = UpSample()
        self.resblocks = torch.nn.ModuleList()
        self.resblocks.append(ResBlock(2 * in_channels, out_channels))

        for _ in range(num_resblocks - 1):
            self.resblocks.append(ResBlock(out_channels, out_channels))

        self.attentionblocks = torch.nn.ModuleList()
        for _ in range(num_resblocks):
            if use_attention:
                self.attentionblocks.append(AttentionBlock(out_channels, num_heads=8))
            else:
                self.attentionblocks.append(torch.nn.Identity())

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)

        for i in range(self.num_resblocks):
            h = self.resblocks[i](h)
            h = self.attentionblocks[i](h)

        return h


class MidBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super(MidBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        self.res_1 = ResBlock(in_channels, out_channels)
        self.res_2 = ResBlock(out_channels, out_channels)

        if use_attention:
            self.attention = AttentionBlock(out_channels, num_heads=8)
        else:
            self.attention = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.res_1(x)
        h = self.attention(h)
        h = self.res_2(h)

        return h


class PadBlock(torch.nn.Module):
    """Padding-Hilfsblock mit (left, right, top, bottom)."""

    def __init__(self, pads: tuple[int, int, int, int]):
        super(PadBlock, self).__init__()

        # pads = (left, right, top, bottom)
        self.pads = pads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, self.pads, mode="constant", value=0)


class UnPadBlock(torch.nn.Module):
    """Entfernt zuvor gesetztes Padding (left, right, top, bottom)."""

    def __init__(self, pads: tuple[int, int, int, int]):
        super(UnPadBlock, self).__init__()

        # pads = (left, right, top, bottom)
        self.pads = pads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, self.pads[2] : -self.pads[3], self.pads[0] : -self.pads[1]]


class UNet(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int,
        channels_mult: list[int],
        num_resblocks: int = 2,
        attention_layers: list[bool],
        pads: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        super(UNet, self).__init__()

        self.in_channels = in_channels  # 69
        self.base_channels = base_channels  # 128
        self.channels_mult = channels_mult  # [1, 1, 2, 2, 4]
        self.num_resblocks = num_resblocks  # 2
        # [False, False, True, True, True]
        self.attention_layers = attention_layers
        self.pads = pads  # (0, 0, 0, 0)

        self.num_layers = len(channels_mult) - 1
        self.channels = [base_channels * mult for mult in channels_mult]

        assert len(channels_mult) == len(attention_layers)

        self.conv_in = torch.nn.Conv2d(
            2 * in_channels, self.channels[0], kernel_size=3, stride=1, padding=1
        )

        self.conv_out = torch.nn.Conv2d(
            self.channels[0], in_channels, kernel_size=3, stride=1, padding=1
        )

        self.pad = PadBlock(pads)
        self.unpad = UnPadBlock(pads)

        self.down_blocks = torch.nn.ModuleList()
        self.mid_blocks = torch.nn.ModuleList()
        self.up_blocks = torch.nn.ModuleList()

        # create down blocks
        for i in range(self.num_layers):
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]

            down_block = DownBlock(
                in_channels,
                out_channels,
                num_resblocks=num_resblocks,
                use_attention=attention_layers[i],
            )

            self.down_blocks.append(down_block)

        # create mid blocks
        self.mid_blocks.append(
            MidBlock(
                self.channels[-1],
                self.channels[-1],
                use_attention=attention_layers[-1],
            )
        )

        # create up blocks
        for i in reversed(range(self.num_layers)):
            in_channels = self.channels[i + 1]
            out_channels = self.channels[i]

            up_block = UpBlock(
                in_channels,
                out_channels,
                num_resblocks=num_resblocks,
                use_attention=attention_layers[i],
            )

            self.up_blocks.append(up_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pad(x)

        h = self.conv_in(h)

        skip_connections = []

        for down_block in self.down_blocks:
            h, skip = down_block(h)
            skip_connections.append(skip)

        h = self.mid_blocks[0](h)

        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            h = up_block(h, skip)

        h = self.conv_out(h)
        h = self.unpad(h)

        return h
