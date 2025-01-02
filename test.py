import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

gs_path = "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr/"

dataset = xr.open_zarr(gs_path, chunks="auto")

z500 = dataset["geopotential"].sel(level=500).drop("level")



z500.to_zarr("z500.zarr", mode="w")


