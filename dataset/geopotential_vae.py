import torch
import lightning.pytorch as pl
import numpy as np
import os
from typing import Optional, Union


class GeopotentialDatasetVAE(torch.utils.data.Dataset):
    def __init__(
        self,
        mode: str = "train",
        transform: Optional[callable] = None,
        data_dir: str = ".",
    ) -> None:
        super().__init__()

        # Set the mode (train, validate, test) and transformation function
        self.mode: str = mode
        self.transform: Optional[callable] = transform
        self.data_dir: str = data_dir

        # Mean and standard deviation for normalization
        self.mean: float = 54028.656
        self.std: float = 3360.1072

        # Load the data using the internal method
        self.data: np.memmap = self._load_data()

    def _load_data(self) -> np.memmap:
        # Determine the file path based on the mode
        if self.mode == "train":
            file_path = f"{self.data_dir}/geopotential_train.memmap"
        elif self.mode == "validate":
            file_path = f"{self.data_dir}/geopotential_validate.memmap"
        elif self.mode == "test":
            file_path = f"{self.data_dir}/geopotential_test.memmap"
        else:
            raise ValueError("Invalid mode! Use 'train', 'validate' or 'test'.")

        # Check if the file exists, raise an error if not
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The file {file_path} does not exist. Please provide a valid file path."
            )

        # Load the data using memory mapping for efficient access
        try:
            data = np.memmap(file_path, mode="r", dtype=np.float32).reshape(-1, 128, 64)
        except ValueError:
            raise ValueError(
                f"The file {file_path} cannot be reshaped to the required dimensions (-1, 128, 64). Please check the file."
            )

        # Normalize the entire dataset
        data = (data - self.mean) / self.std

        return data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple]:
        # Ensure the index is within valid range
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset with length {len(self)}."
            )

        item = torch.from_numpy(self.data[idx]).float()

        # Apply any transformations, if provided
        if self.transform is not None:
            item = self.transform(item)

        return item.unsqueeze(0)

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        # Denormalize the data using the mean and standard deviation
        return data * self.std + self.mean


class GeopotentialDataModuleVAE(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 0) -> None:
        super().__init__()
        # Batch size for data loading
        self.batch_size: int = batch_size
        # Number of worker processes for data loading
        self.num_workers: int = num_workers

        # Set the prefetch factor based on the number of workers
        if self.num_workers <= 0:
            self.prefetch_factor = None
        else:
            self.prefetch_factor = 2

        # Dataset objects will be initialized in the setup method
        self.train_ds: Optional[GeopotentialDatasetVAE] = None
        self.val_ds: Optional[GeopotentialDatasetVAE] = None
        self.test_ds: Optional[GeopotentialDatasetVAE] = None

    def prepare_data(self) -> None:
        # This method can be used to download and prepare data if necessary
        print("Data preparation complete.")

    def setup(self, stage: Optional[str] = None) -> None:
        # Instantiate datasets for training, validation, and testing
        if stage == "fit" or stage is None:
            self.train_ds = GeopotentialDatasetVAE(mode="train", data_dir="./dataset")
            self.val_ds = GeopotentialDatasetVAE(mode="validate", data_dir="./dataset")
        if stage == "test" or stage is None:
            self.test_ds = GeopotentialDatasetVAE(mode="test", data_dir="./dataset")

        print("Data setup complete.")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # Return a DataLoader for the training dataset
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        # Return a DataLoader for the validation dataset
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        # Return a DataLoader for the test dataset
        return torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )


if __name__ == "__main__":
    # Create dataset instances for train, validate, and test modes
    train_ds = GeopotentialDatasetVAE(mode="train", data_dir="./dataset")
    print(f"Number of samples in training dataset: {len(train_ds)}")
    val_ds = GeopotentialDatasetVAE(mode="validate", data_dir="./dataset")
    print(f"Number of samples in validation dataset: {len(val_ds)}")
    test_ds = GeopotentialDatasetVAE(mode="test", data_dir="./dataset")
    print(f"Number of samples in test dataset: {len(test_ds)}")

    # Create an instance of the GeopotentialDataModule
    data = GeopotentialDataModuleVAE()

    item = train_ds[0]
    print(f"Sample item shape: {item.shape}")
