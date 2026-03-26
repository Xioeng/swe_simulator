"""Coordinate mapping between geographic and metric coordinates."""

import numpy as np
import numpy.typing as npt

from .logging_config import get_logger

logger = get_logger(__name__)


class GeographicCoordinateMapper:
    """
    Map between lon/lat (degrees) and x/y (meters) coordinates.

    Parameters
    ----------
    lon0 : float
        Reference longitude in degrees (mapped to x=0)
    lat0 : float
        Reference latitude in degrees (mapped to y=0)
    R : float, default=6371000.0
        Earth radius in meters
    """

    def __init__(self, lon0: float, lat0: float, R: float = 6371000.0) -> None:
        self.lon0: float = lon0
        self.lat0: float = lat0
        self.R: float = R
        # Pre-compute radian versions for conversion methods
        self._lon0_rad: float = np.deg2rad(lon0)
        self._lat0_rad: float = np.deg2rad(lat0)
        self.cos_lat0: float = np.cos(self._lat0_rad)

        logger.debug(
            f"Initialized mapper: lon0={self.lon0:.4f}°, lat0={self.lat0:.4f}°"
        )

    def coord_to_metric(
        self,
        lon: float | npt.NDArray[np.float64],
        lat: float | npt.NDArray[np.float64],
    ) -> tuple[
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
    ]:
        """
        Convert (lon, lat) in degrees to (x, y) in meters.

        Parameters
        ----------
        lon : float or npt.NDArray[np.float64]
            Longitude(s) in degrees
        lat : float or npt.NDArray[np.float64]
            Latitude(s) in degrees

        Returns
        -------
        x : float or npt.NDArray[np.float64]
            X coordinate(s) in meters
        y : float or npt.NDArray[np.float64]
            Y coordinate(s) in meters
        """
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)

        x = self.R * (lon_rad - self._lon0_rad) * self.cos_lat0
        y = self.R * (lat_rad - self._lat0_rad)

        return x, y

    def metric_to_coord(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
    ) -> tuple[
        float | npt.NDArray[np.float64],
        float | npt.NDArray[np.float64],
    ]:
        """
        Convert (x, y) in meters to (lon, lat) in degrees.

        Parameters
        ----------
        x : float or npt.NDArray[np.float64]
            X coordinate(s) in meters
        y : float or npt.NDArray[np.float64]
            Y coordinate(s) in meters

        Returns
        -------
        lon : float or npt.NDArray[np.float64]
            Longitude(s) in degrees
        lat : float or npt.NDArray[np.float64]
            Latitude(s) in degrees
        """
        lon_rad = self._lon0_rad + x / (self.R * self.cos_lat0)
        lat_rad = self._lat0_rad + y / self.R
        lon = np.rad2deg(lon_rad)
        lat = np.rad2deg(lat_rad)
        return lon, lat
