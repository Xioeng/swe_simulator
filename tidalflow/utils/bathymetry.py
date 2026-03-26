"""
Bathymetry utilities for working with GEBCO and other bathymetry data.

This module provides functions for:
- Loading GEBCO NetCDF files
- Building reusable GEBCO interpolators
- Interpolating bathymetry onto simulation grids
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

from ..logging_config import get_logger
from . import grid

logger = get_logger(__name__)


def load_gebco_data(nc_path: str | Path) -> dict[str, npt.NDArray[np.float64]]:
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
            "elevation": dataset.elevation.values.T,  # Transpose to (lon, lat)
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
            "elevation": dataset.variables["elevation"][:].T,  # Transpose to (lon, lat)
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
    nc_path: str | Path,
    method: str = "linear",
) -> RegularGridInterpolator:
    """
    Build a 2D interpolator from GEBCO NetCDF data.

    This creates a reusable interpolator object that can efficiently
    interpolate bathymetry values at arbitrary (lon, lat) points.

    Parameters
    ----------
    nc_path : str or Path
        Path to GEBCO NetCDF file
    method : str, default='linear'
        Interpolation method: 'linear', 'nearest', or 'cubic'

    Returns
    -------
    RegularGridInterpolator
        Interpolator function that takes (lon, lat) points and returns
        elevation values. Call with: interpolator((lon_points, lat_points))

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
    - Handle coordinates in (lon, lat) order
    - Use specified interpolation method
    """
    logger.debug(f"Building GEBCO interpolator with method='{method}'")

    # Load GEBCO data
    gebco_data = load_gebco_data(nc_path)

    # Use generic interpolator builder from grid module
    interpolator = grid.build_regular_grid_interpolator(
        lon=gebco_data["lon"],
        lat=gebco_data["lat"],
        values=gebco_data["elevation"],
        method=method,
    )

    return interpolator


def interpolate_gebco_on_grid(
    X: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
    nc_path: str | Path,
    method: str = "cubic",
    fill_nan_with: float | None = None,
) -> npt.NDArray[np.float64]:
    """
    Interpolate GEBCO bathymetry data onto a 2D grid.

    Convenience function that loads GEBCO data, builds an interpolator,
    and applies it to the provided coordinate grid in one call.

    For repeated interpolations, it is more efficient to build the
    interpolator once via build_gebco_interpolator() and then call
    interpolate_on_mesh() multiple times.

    Parameters
    ----------
    X : np.ndarray
        2D array of longitude coordinates (degrees)
    Y : np.ndarray
        2D array of latitude coordinates (degrees)
    nc_path : str or Path
        Path to GEBCO NetCDF file
    method : str, default='cubic'
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

    See Also
    --------
    build_gebco_interpolator : Build interpolator for reuse on multiple grids
    grid.interpolate_on_mesh : Apply a pre-built interpolator to a mesh grid
    """
    if X.shape != Y.shape:
        raise ValueError(
            f"X and Y must have the same shape. "
            f"Got X.shape={X.shape}, Y.shape={Y.shape}"
        )

    logger.debug(
        f"Interpolating GEBCO data on grid of shape {X.shape} using method='{method}'"
    )

    # Build interpolator
    interpolator = build_gebco_interpolator(nc_path, method=method)

    # Apply to mesh using generic function from grid module
    bathymetry = grid.interpolate_on_mesh(
        interpolator,
        X,
        Y,
        fill_nan_with=fill_nan_with,
    )

    return bathymetry
