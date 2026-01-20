import numpy as np


class LocalLonLatMetricMapper:
    """
    Simple local mapping between lon/lat and x/y in meters,
    assuming a spherical Earth and small domain.
    """

    def __init__(self, lon0: float, lat0: float, R: float = 6371000.0) -> None:
        """
        Parameters
        ----------
        lon0, lat0 : float
            Reference point (deg) mapped to (x=0, y=0).
        R : float
            Earth radius in meters.
        """
        self.lon0: float = np.deg2rad(lon0)
        self.lat0: float = np.deg2rad(lat0)
        self.R: float = R
        self.cos_lat0: float = np.cos(self.lat0)

    def coord_to_metric(
        self, lon: float | np.ndarray, lat: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        (lon,lat) in degrees -> (x,y) in meters, relative to (lon0,lat0).
        Accepts scalars or NumPy arrays.
        """
        lon_rad: float | np.ndarray = np.deg2rad(lon)
        lat_rad: float | np.ndarray = np.deg2rad(lat)

        dlon: float | np.ndarray = lon_rad - self.lon0
        dlat: float | np.ndarray = lat_rad - self.lat0

        x: float | np.ndarray = self.R * dlon * self.cos_lat0
        y: float | np.ndarray = self.R * dlat
        return x, y

    def metric_to_coord(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        (x,y) in meters -> (lon,lat) in degrees.
        Accepts scalars or NumPy arrays.
        """
        lon_rad: float | np.ndarray = x / (self.R * self.cos_lat0) + self.lon0
        lat_rad: float | np.ndarray = y / self.R + self.lat0
        lon: float | np.ndarray = np.rad2deg(lon_rad)
        lat: float | np.ndarray = np.rad2deg(lat_rad)
        return lon, lat
