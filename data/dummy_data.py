import torch
import lightning as pl


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, config, size):
        self.config = config
        self.size = size
        self.data = torch.randn(
            self.config["config"]["data"]["in_channels"],
            self.config["config"]["data"]["x"],
            self.config["config"]["data"]["y"],
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, config, size):
        super().__init__()
        self.config = config
        self.size = size

    def setup(self, stage=None):
        self.train_dataset = DummyDataset(self.config, self.size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["config"]["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["config"]["dataloader"]["num_workers"],
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["config"]["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["config"]["dataloader"]["num_workers"],
            persistent_workers=True,
        )


def main():
    import yaml

    with open("configs/autoencoder_kl_f8.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_module = DummyDataModule(config, 100000)
    data_module.setup()

    for batch in data_module.train_dataloader():
        print(batch.shape)


if __name__ == "__main__":
    main()
