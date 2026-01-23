"""
Bathymetry utilities for working with GEBCO and other bathymetry data.

This module provides functions for:
- Loading GEBCO NetCDF files
- Interpolating bathymetry onto simulation grids
- Building reusable interpolators
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


def load_gebco_data(nc_path: Union[str, Path]) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Load GEBCO netCDF bathymetry data.

    This function reads GEBCO (General Bathymetric Chart of the Oceans) data
    from a NetCDF file and returns the coordinates and elevation values.

    Parameters
    ----------
    nc_path : str or Path
        Path to GEBCO netCDF file (*.nc)

    Returns
    -------
    dict
        Dictionary containing:
        - 'lon': 1D array of longitude values (degrees)
        - 'lat': 1D array of latitude values (degrees)
        - 'elevation': 2D array of elevation values (meters)
          Negative values indicate depth below sea level.

    Raises
    ------
    FileNotFoundError
        If the netCDF file doesn't exist
    ImportError
        If neither xarray nor netCDF4 is installed
    ValueError
        If the file doesn't contain required variables

    Notes
    -----
    This function prefers xarray if available, falling back to netCDF4.
    GEBCO elevation values follow the convention:
    - Positive values: elevation above sea level
    - Negative values: depth below sea level (bathymetry)
    """
    nc_path = Path(nc_path)

    if not nc_path.exists():
        raise FileNotFoundError(f"GEBCO file not found: {nc_path}")

    # Try xarray first (preferred)
    try:
        import xarray as xr

        logger.debug(f"Loading GEBCO data using xarray from: {nc_path}")
        dataset = xr.open_dataset(nc_path)

        # Check for required variables
        required_vars = ["lon", "lat", "elevation"]
        missing = [var for var in required_vars if var not in dataset.variables]
        if missing:
            raise ValueError(
                f"GEBCO file missing required variables: {missing}. "
                f"Available: {list(dataset.variables.keys())}"
            )

        data = {
            "lon": dataset.lon.values,
            "lat": dataset.lat.values,
            "elevation": dataset.elevation.values,
        }

        logger.info(
            f"Loaded GEBCO data: lon=[{data['lon'].min():.2f}, {data['lon'].max():.2f}], "
            f"lat=[{data['lat'].min():.2f}, {data['lat'].max():.2f}], "
            f"elevation=[{data['elevation'].min():.2f}, {data['elevation'].max():.2f}]m"
        )

        return data

    except ImportError:
        pass  # Fall back to netCDF4

    # Try netCDF4 as fallback
    try:
        import netCDF4 as nc

        logger.debug(f"Loading GEBCO data using netCDF4 from: {nc_path}")
        dataset = nc.Dataset(nc_path, "r")

        # Check for required variables
        required_vars = ["lon", "lat", "elevation"]
        missing = [var for var in required_vars if var not in dataset.variables]
        if missing:
            raise ValueError(
                f"GEBCO file missing required variables: {missing}. "
                f"Available: {list(dataset.variables.keys())}"
            )

        data = {
            "lon": dataset.variables["lon"][:],
            "lat": dataset.variables["lat"][:],
            "elevation": dataset.variables["elevation"][:],
        }

        dataset.close()

        logger.info(
            f"Loaded GEBCO data: lon=[{data['lon'].min():.2f}, {data['lon'].max():.2f}], "
            f"lat=[{data['lat'].min():.2f}, {data['lat'].max():.2f}], "
            f"elevation=[{data['elevation'].min():.2f}, {data['elevation'].max():.2f}]m"
        )

        return data

    except ImportError:
        raise ImportError(
            "Either xarray or netCDF4 is required to read GEBCO data.\n"
            "Install with: pip install xarray  or  pip install netCDF4"
        )


def build_gebco_interpolator(
    nc_path: Union[str, Path],
    method: str = "linear",
) -> RegularGridInterpolator:
    """
    Build a 2D interpolator from GEBCO NetCDF data.

    This creates a reusable interpolator object that can efficiently
    interpolate bathymetry values at arbitrary (lat, lon) points.

    Parameters
    ----------
    nc_path : str or Path
        Path to GEBCO NetCDF file
    method : str, default='linear'
        Interpolation method: 'linear', 'nearest', or 'cubic'

    Returns
    -------
    RegularGridInterpolator
        Interpolator function that takes (lat, lon) points and returns
        elevation values. Call with: interpolator((lat_points, lon_points))

    Raises
    ------
    FileNotFoundError
        If GEBCO file doesn't exist
    ValueError
        If interpolation method is invalid

    Notes
    -----
    The interpolator is configured to:
    - Return NaN for points outside the data bounds (bounds_error=False)
    - Handle coordinates in (lat, lon) order
    - Use specified interpolation method
    """
    valid_methods = ["linear", "nearest", "cubic"]
    if method not in valid_methods:
        raise ValueError(
            f"Invalid interpolation method: {method}. Must be one of {valid_methods}"
        )

    logger.debug(f"Building GEBCO interpolator with method='{method}'")

    # Load GEBCO data
    gebco_data = load_gebco_data(nc_path)

    lon = gebco_data["lon"]
    lat = gebco_data["lat"]
    elevation = gebco_data["elevation"]

    # Ensure latitude is in increasing order (required by RegularGridInterpolator)
    if lat[1] < lat[0]:
        logger.debug("Reversing latitude array to ensure increasing order")
        lat = lat[::-1]
        elevation = elevation[::-1, :]

    # Create interpolator (lat, lon) -> elevation
    interpolator = RegularGridInterpolator(
        (lat, lon),
        elevation,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    logger.info(
        f"Created GEBCO interpolator: "
        f"lat=[{lat.min():.2f}, {lat.max():.2f}], "
        f"lon=[{lon.min():.2f}, {lon.max():.2f}], "
        f"method='{method}'"
    )

    return interpolator


def interpolate_gebco_on_grid(
    X: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
    nc_path: Union[str, Path],
    method: str = "linear",
    fill_nan_with: Optional[float] = None,
) -> npt.NDArray[np.float64]:
    """
    Interpolate GEBCO bathymetry data onto a 2D grid.

    This is a convenience function that loads GEBCO data, builds an
    interpolator, and applies it to the provided coordinate grid.

    Parameters
    ----------
    X : np.ndarray
        2D array of longitude coordinates (degrees)
    Y : np.ndarray
        2D array of latitude coordinates (degrees)
    nc_path : str or Path
        Path to GEBCO NetCDF file
    method : str, default='linear'
        Interpolation method: 'linear', 'nearest', or 'cubic'
    fill_nan_with : float, optional
        Value to replace NaN values with. If None, NaNs are preserved.

    Returns
    -------
    np.ndarray
        2D array of interpolated elevation values (meters) matching
        the shape of X and Y. Negative values indicate depth below sea level.

    Raises
    ------
    ValueError
        If X and Y have different shapes

    Notes
    -----
    - Points outside the GEBCO data bounds will be NaN (unless fill_nan_with is set)
    - For repeated interpolations on the same GEBCO data, consider using
      build_gebco_interpolator() once and reusing it for better performance
    """
    if X.shape != Y.shape:
        raise ValueError(
            f"X and Y must have the same shape. Got X.shape={X.shape}, Y.shape={Y.shape}"
        )

    logger.debug(
        f"Interpolating GEBCO data on grid of shape {X.shape} using method='{method}'"
    )

    # Build interpolator
    interpolator = build_gebco_interpolator(nc_path, method=method)

    # Prepare points for interpolation (lat, lon) pairs
    points = np.column_stack([Y.ravel(), X.ravel()])

    # Interpolate
    bathymetry = interpolator(points).reshape(X.shape)

    # Count NaN values
    n_nan = np.sum(np.isnan(bathymetry))
    if n_nan > 0:
        logger.warning(
            f"Interpolation resulted in {n_nan} NaN values "
            f"({100 * n_nan / bathymetry.size:.2f}% of grid points)"
        )

        if fill_nan_with is not None:
            logger.info(f"Filling NaN values with {fill_nan_with}")
            bathymetry = np.nan_to_num(bathymetry, nan=fill_nan_with)

    logger.info(
        f"Interpolated bathymetry: min={np.nanmin(bathymetry):.2f}m, "
        f"max={np.nanmax(bathymetry):.2f}m, "
        f"mean={np.nanmean(bathymetry):.2f}m"
    )

    return bathymetry
