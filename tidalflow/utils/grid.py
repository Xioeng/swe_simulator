"""
Grid utilities for mesh generation and interpolation.

This module provides functions for:
- Generating mesh grids (cell centers)
- Building general-purpose interpolators for gridded data
- Applying interpolators to mesh grids
"""

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)

from ..logging_config import get_logger

logger = get_logger(__name__)


def generate_cell_centers(
    x_lower: float, x_upper: float, y_lower: float, y_upper: float, nx: int, ny: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    X, Y : npt.NDArray[np.float64]
        2D arrays of cell center coordinates
    """
    dx = (x_upper - x_lower) / nx
    dy = (y_upper - y_lower) / ny

    x_centers = x_lower + (np.arange(nx) + 0.5) * dx
    y_centers = y_lower + (np.arange(ny) + 0.5) * dy

    X, Y = np.meshgrid(
        x_centers, y_centers, indexing="ij"
    )  # We want 'ij' for (nx, ny) shape for pyclaw compatibility
    return X, Y


def build_regular_grid_interpolator(
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    method: str = "linear",
) -> RegularGridInterpolator:
    """
    Build a 2D interpolator from regular gridded data.

    This is a generic interpolator factory for any 2D gridded data
    (e.g., bathymetry, temperature, salinity).

    Parameters
    ----------
    lon : npt.NDArray[np.float64]
        1D array of longitude coordinates (degrees)
    lat : npt.NDArray[np.float64]
        1D array of latitude coordinates (degrees)
    values : npt.NDArray[np.float64]
        2D array of values on the (lat, lon) grid
    method : str, default='linear'
        Interpolation method: 'linear', 'nearest', or 'cubic'

    Returns
    -------
    RegularGridInterpolator
        Interpolator that maps (lon, lat) points to interpolated values.
        Call with: interpolator((lon_points, lat_points))

    Raises
    ------
    ValueError
        If method is invalid or array shapes are incompatible

    Notes
    -----
        - Ensures longitude and latitude are in increasing order
            (required by RegularGridInterpolator)
    - Returns NaN for points outside data bounds
    - Assumes values shape is (len(lon), len(lat))
    """
    valid_methods = ["linear", "nearest", "cubic"]
    if method not in valid_methods:
        raise ValueError(
            f"Invalid interpolation method: {method}. Must be one of {valid_methods}"
        )

    if values.shape != (len(lon), len(lat)):
        raise ValueError(
            f"values shape {values.shape} does not match "
            f"(len(lon), len(lat)) = ({len(lon)}, {len(lat)})"
        )

    logger.debug(
        f"Building regular grid interpolator: shape={values.shape}, method='{method}'"
    )

    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)

    # Ensure longitude is in increasing order
    # (required by RegularGridInterpolator)
    if len(lon_arr) > 1 and lon_arr[1] < lon_arr[0]:
        logger.debug("Reversing longitude array to ensure increasing order")
        lon_arr = lon_arr[::-1]
        values_arr = values_arr[:, ::-1]

    # Ensure latitude is in increasing order
    # (required by RegularGridInterpolator)
    if len(lat_arr) > 1 and lat_arr[1] < lat_arr[0]:
        logger.debug("Reversing latitude array to ensure increasing order")
        lat_arr = lat_arr[::-1]
        values_arr = values_arr[::-1, :]

    # Regular-grid values are provided as (lon, lat).
    # Create and return interpolator
    interpolator = RegularGridInterpolator(
        (lon_arr, lat_arr),
        values_arr,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    logger.info(
        f"Created interpolator: "
        f"lat=[{lat_arr.min():.2f}, {lat_arr.max():.2f}], "
        f"lon=[{lon_arr.min():.2f}, {lon_arr.max():.2f}], "
        f"method='{method}'"
    )

    return interpolator


def build_scattered_interpolator(
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    method: str = "linear",
    fill_value: float = np.nan,
    use_nearest_fallback: bool = True,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """
    Build an interpolator from scattered (lon, lat, value) samples.

    Parameters
    ----------
    lon : npt.NDArray[np.float64]
        1D array of longitudes for each sample point.
    lat : npt.NDArray[np.float64]
        1D array of latitudes for each sample point.
    values : npt.NDArray[np.float64]
        1D array of values associated with each (lon, lat) point.
    method : str, default='linear'
        Interpolation method: 'linear' or 'nearest'.
    fill_value : float, default=np.nan
        Fill value used by linear interpolation outside the convex hull.
    use_nearest_fallback : bool, default=True
        If True with method='linear', replace NaN outputs with nearest-neighbor
        interpolation values.

    Returns
    -------
    Callable
        Callable interpolator expecting points in (lon, lat) order with shape
        (n_points, 2), consistent with interpolate_on_mesh().

    Raises
    ------
    ValueError
        If inputs do not have matching shapes or method is invalid.
    """
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'")

    lon_arr = np.asarray(lon, dtype=np.float64).ravel()
    lat_arr = np.asarray(lat, dtype=np.float64).ravel()
    values_arr = np.asarray(values, dtype=np.float64).ravel()

    if lon_arr.shape != lat_arr.shape or lon_arr.shape != values_arr.shape:
        raise ValueError(
            "lon, lat, and values must have the same 1D shape. "
            f"Got lon={lon_arr.shape}, lat={lat_arr.shape}, values={values_arr.shape}"
        )

    # Interpolators in this module use (lon, lat) coordinate order.
    points = np.column_stack([lon_arr, lat_arr])

    if method == "nearest":
        nearest_interpolator = NearestNDInterpolator(points, values_arr)
        return cast(
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
            nearest_interpolator,
        )

    linear_interpolator = LinearNDInterpolator(
        points,
        values_arr,
        fill_value=fill_value,
    )

    if not use_nearest_fallback:
        return cast(
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
            linear_interpolator,
        )

    nearest_interpolator = NearestNDInterpolator(points, values_arr)

    def interpolator(query_points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        linear_values = np.asarray(linear_interpolator(query_points), dtype=np.float64)
        nan_mask = np.isnan(linear_values)
        if np.any(nan_mask):
            nearest_values = np.asarray(
                nearest_interpolator(query_points),
                dtype=np.float64,
            )
            linear_values[nan_mask] = nearest_values[nan_mask]
        return linear_values

    return interpolator


def interpolate_on_mesh(
    interpolator: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    X: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
    fill_nan_with: float | None = None,
) -> npt.NDArray[np.float64]:
    """
    Apply an interpolator to a mesh grid.

    This is a generic function that works with any interpolator callable
    expecting query points in (lon, lat) order.

    Parameters
    ----------
    interpolator : Callable
        Pre-built interpolator callable (from build_regular_grid_interpolator,
        build_scattered_interpolator, build_gebco_interpolator, etc.)
    X : npt.NDArray[np.float64]
        2D array of longitude coordinates (degrees)
    Y : npt.NDArray[np.float64]
        2D array of latitude coordinates (degrees)
    fill_nan_with : float, optional
        Value to replace NaN values with. If None, NaNs are preserved.

    Returns
    -------
    npt.NDArray[np.float64]
        2D array of interpolated values matching the shape of X and Y.

    Raises
    ------
    ValueError
        If X and Y have different shapes

    Notes
    -----
    - Points outside the interpolator's data bounds will be NaN
      (unless fill_nan_with is set)
        - Assumes interpolator uses (lon, lat) coordinate order
    """
    if X.shape != Y.shape:
        raise ValueError(
            f"X and Y must have the same shape. "
            f"Got X.shape={X.shape}, Y.shape={Y.shape}"
        )

    logger.debug(f"Interpolating on mesh of shape {X.shape}")

    # Prepare points for interpolation (lon, lat) pairs
    points = np.column_stack([X.ravel(), Y.ravel()])

    # Interpolate
    result: npt.NDArray[np.float64] = interpolator(points).reshape(X.shape)

    # Handle NaN values
    n_nan = np.sum(np.isnan(result))
    if n_nan > 0:
        logger.warning(
            f"Interpolation resulted in {n_nan} NaN values "
            f"({100 * n_nan / result.size:.2f}% of grid points)"
        )

        if fill_nan_with is not None:
            logger.info(f"Filling NaN values with {fill_nan_with}")
            result = np.nan_to_num(result, nan=fill_nan_with)

    logger.info(
        f"Interpolated mesh: min={np.nanmin(result):.2f}, "
        f"max={np.nanmax(result):.2f}, "
        f"mean={np.nanmean(result):.2f}"
    )

    return result
