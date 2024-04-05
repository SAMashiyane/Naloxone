

import numpy as np
import scikitplot as skplt


def leverage_statistic(x: np.ndarray):

    if x.ndim == 1:
        x = x.reshape(x.shape[0], 1)

    cov_mat_inv = np.linalg.inv(x.T.dot(x))
    H = x.dot(cov_mat_inv.dot(x.T))
    leverage = H.diagonal()
    return leverage


def calculate_standardized_residual(
    predicted: np.ndarray, expected: np.ndarray = None, featuresize: int = None
) -> np.ndarray:

    if expected is not None:
        residuals = expected - predicted
    else:
        residuals = predicted
        expected = predicted

    if residuals.sum() == 0:
        return residuals

    n = residuals.shape[0]
    m = featuresize
    if m is None:
        m = 1
    s2_hat = 1 / (n - m) * np.sum(residuals**2)
    leverage = 1 / n + (expected - np.mean(expected)) / np.sum(
        (expected - np.mean(expected)) ** 2
    )
    standardized_residuals = residuals / (np.sqrt(s2_hat) * (1 - leverage))
    return standardized_residuals


def cooks_distance(
    standardized_residuals: np.ndarray,
    leverage_statistic: np.ndarray,
    n_model_params: int = None,
) -> np.array:

    p = n_model_params if n_model_params is not None and n_model_params >= 1 else 1
    multiplier = [element / (1 - element) for element in leverage_statistic]
    distance = np.multiply(np.power(standardized_residuals, 2) / (p + 1), multiplier)
    return distance


class MatplotlibDefaultDPI(object):
    def __init__(self, base_dpi: float = 100, scale_to_set: float = 1):
        try:
            self.default_skplt_dpit = skplt.metrics.plt.rcParams["figure.dpi"]
            skplt.metrics.plt.rcParams["figure.dpi"] = base_dpi * scale_to_set
        except Exception:
            pass

    def __enter__(self) -> None:
        return None

    def __exit__(self, type, value, traceback):
        try:
            skplt.metrics.plt.rcParams["figure.dpi"] = self.default_skplt_dpit
        except Exception:
            pass
