"""Abstract base classes for providing simulation inputs to the solver."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class InitialConditionProvider(ABC):
    """
    Abstract base class for providing initial conditions to the solver.

    Subclasses must implement the get_initial_condition method to return
    the shallow water equation variables (h, hu, hv) on the simulation grid.
    """

    @abstractmethod
    def get_initial_condition(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Generate or load initial conditions for the simulation.

        Parameters
        ----------
        lon : np.ndarray
            Longitude coordinates meshgrid of shape (ny, nx)
        lat : np.ndarray
            Latitude coordinates meshgrid of shape (ny, nx)

        Returns
        -------
        np.ndarray
            Array of shape (3, ny, nx) containing:
            - h: water depth
            - hu: x-momentum (depth × u-velocity)
            - hv: y-momentum (depth × v-velocity)
        """
        pass


class WindProvider(ABC):
    """
    Abstract base class for providing wind forcing as a function of time.

    Subclasses must implement the get_wind method to return wind velocities
    at a given simulation time.
    """

    @abstractmethod
    def get_wind(
        self, lon: npt.NDArray[np.float64], lat: npt.NDArray[np.float64], time: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get wind velocities at a given time.

        Parameters
        ----------
        lon : np.ndarray
            Longitude coordinates meshgrid of shape (ny, nx)
        lat : np.ndarray
            Latitude coordinates meshgrid of shape (ny, nx)
        time : float
            Current simulation time in seconds

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Wind velocities (u_wind, v_wind) in m/s
        """
        pass


class BathymetryProvider(ABC):
    """
    Abstract base class for providing bathymetry (seabed elevation).

    Subclasses must implement the get_bathymetry method to return
    the bathymetry on the simulation grid.
    """

    @abstractmethod
    def get_bathymetry(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Generate or load bathymetry data.

        Parameters
        ----------
        lon : np.ndarray
            Longitude coordinates meshgrid of shape (ny, nx)
        lat : np.ndarray
            Latitude coordinates meshgrid of shape (ny, nx)

        Returns
        -------
        np.ndarray
            Array of shape (ny, nx) containing bathymetry (depth) values in meters.
            Negative values represent depth below sea level.
        """
        pass
