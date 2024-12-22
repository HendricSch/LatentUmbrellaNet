import os
import xarray as xr
import numpy as np


def is_datasets_already_created(dataset_name: str) -> bool:
    """
    Check if the dataset is already created
    :param dataset_name: Name of the dataset
    :return: True if the dataset is already created, False otherwise
    """
    test = os.path.exists(f"dataset/{dataset_name}_test.memmap")
    train = os.path.exists(f"dataset/{dataset_name}_train.memmap")
    val = os.path.exists(f"dataset/{dataset_name}_validate.memmap")

    return test and train and val


def load_dataset_from_gs(gs_path: str) -> xr.Dataset:
    """
    Load a dataset from Google Storage
    :param gs_path: Path to the dataset in Google Storage
    :return: The dataset
    """
    return xr.open_zarr(gs_path)


def save_dataset_to_zarr(dataset: xr.Dataset, dataset_name: str) -> None:
    """
    Save a dataset to a Zarr file
    :param dataset: The dataset
    :param dataset_name: Name of the dataset
    """
    dataset.to_zarr(f"dataset/{dataset_name}.zarr")


def filter_dataset(
    dataset: xr.Dataset, data_variables: list[tuple[str, int]]
) -> xr.Dataset:
    """
    Filter the dataset to keep only the specified data variables
    :param dataset: The dataset
    :param data_variables: List of data variables to keep
    :return: The filtered dataset
    """

    # the first element of the tuple is the variable name and the second element is the level

    for data_var, level in data_variables:
        if data_var not in dataset:
            raise ValueError(f"Variable {data_var} not found in the dataset.")
        if "level" in dataset[data_var].dims:
            if level not in dataset[data_var]["level"]:
                raise ValueError(f"Level {level} not found in variable {data_var}.")
        dataset = dataset[data_var].sel(level=level).drop("level")

    return dataset


def split_dataset(
    dataset: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Split the dataset into training, validation and testing datasets
    :param dataset: The dataset
    :return: A tuple containing the training, validation and testing datasets
    """

    da_train = dataset.sel(time=slice("1959-01-01", "2009-12-31"))
    da_val = dataset.sel(time=slice("2010-01-01", "2015-12-31"))
    da_test = dataset.sel(time=slice("2016-01-01", "2021-12-31"))

    return da_train, da_val, da_test


def save_dataset_to_memmap(
    train: np.ndarray, val: np.ndarray, test: np.ndarray, dataset_name: str
) -> None:
    """
    Save a dataset to a memmap file
    :param dataset: The dataset
    :param dataset_name: Name of the dataset
    """

    data_memmap_train = np.memmap(
        f"dataset/{dataset_name}_train.memmap",
        mode="w+",
        shape=train.shape,
        dtype=np.float32,
    )
    data_memmap_train[:] = train[:]
    del data_memmap_train

    data_memmap_val = np.memmap(
        f"dataset/{dataset_name}_validate.memmap",
        mode="w+",
        shape=val.shape,
        dtype=np.float32,
    )
    data_memmap_val[:] = val[:]
    del data_memmap_val

    data_memmap_test = np.memmap(
        f"dataset/{dataset_name}_test.memmap",
        mode="w+",
        shape=test.shape,
        dtype=np.float32,
    )
    data_memmap_test[:] = test[:]
    del data_memmap_test


def create_datasets(
    dataset_name: str, gs_path: str, data_variables: list[tuple[str, int]]
) -> None:
    # Check if the dataset is already created
    if is_datasets_already_created(dataset_name):
        print(f"Dataset {dataset_name} is already created.")
        return

    print(f"Dataset {dataset_name} is not created yet or is incomplete.")
    print("Creating dataset...")

    # Load the dataset from Google Storage
    print("Loading dataset from Google Storage...")
    dataset = load_dataset_from_gs(gs_path)

    # Filter the dataset
    print("Filtering the dataset...")
    dataset = filter_dataset(dataset, data_variables)

    # Save the dataset to Zarr file
    print("Download and saving the dataset to Zarr file...")
    save_dataset_to_zarr(dataset, dataset_name)

    # Load the dataset from Zarr file
    print("Loading the dataset from Zarr file...")
    dataset = xr.open_zarr(f"dataset/{dataset_name}.zarr")

    # Split the dataset
    print("Splitting the dataset...")
    da_train, da_val, da_test = split_dataset(dataset)

    # convert to numpy array
    print("Converting to numpy array...")
    train = da_train.values
    val = da_val.values
    test = da_test.values

    # Save the dataset to memmap files
    print("Saving the dataset to memmap files...")
    save_dataset_to_memmap(train, val, test, dataset_name)


def main() -> None:
    dataset_name = "geopotential2"
    gs_path = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr/"
    data_variables = [("geopotential", 500)]

    create_datasets(
        dataset_name=dataset_name, gs_path=gs_path, data_variables=data_variables
    )


if __name__ == "__main__":
    main()
