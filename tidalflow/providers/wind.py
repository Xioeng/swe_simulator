"""Concrete implementations of WindProvider."""

import numpy as np
import numpy.typing as npt

from .base import WindProvider


class ConstantWind(WindProvider):
    """Constant wind velocity (independent of time)."""

    def __init__(self, u_wind: float = 0.0, v_wind: float = 0.0):
        """
        Create constant wind forcing.

        Parameters
        ----------
        u_wind : float, default=0.0
            Constant x-component of wind (m/s)
        v_wind : float, default=0.0
            Constant y-component of wind (m/s)
        """
        self.u_wind = u_wind
        self.v_wind = v_wind

    def get_wind(
        self, lon, lat, time: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return constant wind regardless of time.

        Parameters
        ----------
        time : float
            Current simulation time (unused)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Wind velocities (u_wind, v_wind)
        """
        return (self.u_wind * np.ones_like(lon), self.v_wind * np.ones_like(lat))
