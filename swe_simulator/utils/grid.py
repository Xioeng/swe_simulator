from typing import Tuple

import numpy as np


def generate_cell_centers(
    x_lower: float, x_upper: float, y_lower: float, y_upper: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cell center coordinates for a 2D grid.

    Parameters
    ----------
    x_lower, x_upper : float
        X domain bounds
    y_lower, y_upper : float
        Y domain bounds
    nx, ny : int
        Number of cells in each direction

    Returns
    -------
    X, Y : np.ndarray
        2D arrays of cell center coordinates
    """
    dx = (x_upper - x_lower) / nx
    dy = (y_upper - y_lower) / ny

    x_centers = x_lower + (np.arange(nx) + 0.5) * dx
    y_centers = y_lower + (np.arange(ny) + 0.5) * dy

    X, Y = np.meshgrid(x_centers, y_centers)
    return X, Y
