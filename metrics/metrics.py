import numpy as np


class WeightedRMSE:
    """RMSE mit breitengrad-gewichteter Flächenmittelung (NumPy-Variante)."""

    def __init__(self, num_latitudes: int):
        self.num_latitudes = num_latitudes
        self.weights = self._get_lat_weights(num_latitudes)

    def __call__(self, truth: np.ndarray, forecast: np.ndarray) -> np.ndarray:
        if truth.shape != forecast.shape:
            raise ValueError("Input arrays must have the same shape.")

        if len(truth.shape) < 3:
            raise ValueError(
                "Input arrays must have at least two dimensions and one feature channel (c x longitude x latitude) or mini-batched (batchsize x c x longitude x latitude)."
            )

        if len(truth.shape) == 3:
            _, _, lat = truth.shape[0], truth.shape[1], truth.shape[2]

        if len(truth.shape) == 4:
            _, _, _, lat = (
                truth.shape[0],
                truth.shape[1],
                truth.shape[2],
                truth.shape[3],
            )

        if lat != self.num_latitudes:
            raise ValueError(
                f"Number of latitudes ({lat}) must match the number of latitudes used to calculate the weights ({self.num_latitudes})."
            )

        return self._spatial_average_l2_norm(truth - forecast)

    def _latitude_cell_bounds(self, x: np.ndarray) -> np.ndarray:
        pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
        return np.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])

    def _cell_area_from_latitude(self, points: np.ndarray) -> np.ndarray:
        bounds = self._latitude_cell_bounds(points)
        upper = bounds[1:]
        lower = bounds[:-1]
        return np.sin(upper) - np.sin(lower)

    def _get_lat_weights(self, num_latitudes: int = 121) -> np.ndarray:
        weights = self._cell_area_from_latitude(
            np.deg2rad(np.linspace(-90, 90, num_latitudes))
        )
        weights /= np.mean(weights)
        return weights

    def _spatial_average(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) < 3:
            raise ValueError(
                "Input array must have at least two dimensions and one feature channel (c x longitude x latitude) or mini-batched (batchsize x c x longitude x latitude)."
            )

        if len(x.shape) == 3:
            c, long, lat = x.shape[0], x.shape[1], x.shape[2]

        if len(x.shape) == 4:
            batchsize, c, long, lat = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        x = x * self.weights

        if len(x.shape) == 3:
            return np.mean(x, axis=(1, 2))

        if len(x.shape) == 4:
            return np.mean(x, axis=(2, 3))

    def _spatial_average_l2_norm(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(self._spatial_average(x**2))
