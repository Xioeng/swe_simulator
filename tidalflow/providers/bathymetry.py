"""Concrete implementations of BathymetryProvider."""

import functools
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from .. import utils
from .base import BathymetryProvider


class FlatBathymetry(BathymetryProvider):
    """Uniform depth bathymetry."""

    def __init__(self, depth: float = -10.0):
        """
        Create flat bathymetry.

        Parameters
        ----------
        depth : float, default=-10.0
            Uniform depth in meters (negative below sea level)
        """
        self.depth = depth

    def get_bathymetry(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return array with uniform depth.

        Parameters
        ----------
        lon : np.ndarray
            Longitude meshgrid of shape (ny, nx)
        lat : np.ndarray
            Latitude meshgrid of shape (ny, nx)

        Returns
        -------
        np.ndarray
            Array of shape (ny, nx) with constant depth values
        """
        return self.depth * np.ones_like(lon)


class SlopingBathymetry(BathymetryProvider):
    """Bathymetry that slopes gradually in one direction."""

    def __init__(self, depth_min: float = -5.0, depth_max: float = -20.0):
        """
        Create sloping bathymetry.

        Parameters
        ----------
        depth_min : float, default=-5.0
            Shallowest depth (m), at y=0
        depth_max : float, default=-20.0
            Deepest depth (m), at y=max
        """
        self.depth_min = depth_min
        self.depth_max = depth_max

    def get_bathymetry(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return linearly sloping bathymetry in y-direction.

        Parameters
        ----------
        lon : np.ndarray
            Longitude meshgrid of shape (ny, nx)
        lat : np.ndarray
            Latitude meshgrid of shape (ny, nx)

        Returns
        -------
        np.ndarray
            Array of shape (ny, nx) with linearly varying depth
        """
        # Normalize latitude to [0, 1]
        lat_min, lat_max = np.min(lat), np.max(lat)
        lat_normalized = (lat - lat_min) / (lat_max - lat_min)

        bathymetry = self.depth_min + (self.depth_max - self.depth_min) * lat_normalized
        return bathymetry


class BathymetryFromNC(BathymetryProvider):
    """Bathymetry loaded from a NetCDF file using interpolation."""

    def __init__(self, nc_path: str | Path):
        """
        Create bathymetry provider from a NetCDF file.

        Parameters
        ----------
        nc_path : str | Path
            Path to the NetCDF file containing bathymetry data
        """
        self.nc_path = Path(nc_path)

        self.bathymetry_interpolator = utils.bathymetry.build_gebco_interpolator(
            nc_path=self.nc_path,
            method="cubic",
        )

    def get_bathymetry(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return bathymetry interpolated from NetCDF file.

        Parameters
        ----------
        lon : np.ndarray
            Longitude meshgrid of shape (ny, nx)
        lat : np.ndarray
            Latitude meshgrid of shape (ny, nx)

        Returns
        -------
        np.ndarray
            Array of shape (ny, nx) with bathymetry values from file
        """
        bathymetry_values = utils.grid.interpolate_on_mesh(
            self.bathymetry_interpolator,
            lon,
            lat,
            fill_nan_with=0.0,
        )
        return bathymetry_values


class BathymetryFromCSV(BathymetryProvider):
    """Bathymetry loaded from a CSV file with columns for lon, lat, elevation."""

    def __init__(
        self,
        csv_path: str | Path,
        columns: tuple[str, str, str] = ("lon", "lat", "elevation"),
        method: str = "linear",
    ):
        """
        Create bathymetry provider from a CSV file.

        Parameters
        ----------
        csv_path : str | Path
            Path to the CSV file containing bathymetry data with columns
            'lon', 'lat', 'elevation'
        columns : tuple[str, str, str], default=("lon", "lat", "elevation")
            Column names for longitude, latitude, and elevation data
        method : str, default='linear'
            Interpolation method for scattered data ('linear' or 'nearest').
        """
        self.csv_path = Path(csv_path)
        lon_col, lat_col, elevation_col = columns

        # Read CSV data
        df = pd.read_csv(self.csv_path)
        lon_data = df[lon_col].to_numpy(dtype=np.float64)
        lat_data = df[lat_col].to_numpy(dtype=np.float64)
        elevation_data = df[elevation_col].to_numpy(dtype=np.float64)

        # Always treat CSV input as unstructured (scattered) triples.
        self.bathymetry_interpolator = utils.grid.build_scattered_interpolator(
            lon=lon_data,
            lat=lat_data,
            values=elevation_data,
            method=method,
            use_nearest_fallback=True,
        )

    def get_bathymetry(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return bathymetry interpolated from CSV file.

        Parameters
        ----------
        lon : np.ndarray
            Longitude meshgrid of shape (ny, nx)
        lat : np.ndarray
            Latitude meshgrid of shape (ny, nx)

        Returns
        -------
        np.ndarray
            Array of shape (ny, nx) with bathymetry values from file
        """
        bathymetry_values = utils.grid.interpolate_on_mesh(
            self.bathymetry_interpolator, lon, lat, 0.0
        )
        return bathymetry_values
