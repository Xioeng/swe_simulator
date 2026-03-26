"""Concrete implementations of InitialConditionProvider."""

import numpy as np
import numpy.typing as npt

from .base import InitialConditionProvider


class GaussianHumpInitialCondition(InitialConditionProvider):
    """Gaussian hump initial condition."""

    def __init__(
        self,
        height: float = 2.0,
        width: float = 0.01,
        bias: float = 0.0,
        center: tuple[float, float] = (0.0, 0.0),
        water_velocity: tuple[float, float] = (0.0, 0.0),
    ):
        """
        Create a Gaussian hump initial condition.

        Parameters
        ----------
        height : float, default=2.0
            Height of the Gaussian hump
        width : float, default=0.01
            Width parameter (controls spread)
        center : tuple[float, float], default=(0.0, 0.0)
            Center of the Gaussian hump
        """
        self.height = height
        self.width = width
        self.center = center
        self.bias = bias
        self.water_velocity = water_velocity

    def get_initial_condition(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate Gaussian hump centered at domain origin.

        Returns (3, ny, nx) array with [h, hu, hv] where:
        - h: Gaussian profile
        - hu, hv: zero (no initial velocity)
        """
        # Gaussian profile using lon/lat coordinates
        h = self.bias + self.height * np.exp(
            -self.width * ((lon - self.center[0]) ** 2 + (lat - self.center[1]) ** 2)
        )
        hu = self.water_velocity[0] * np.ones_like(h)
        hv = self.water_velocity[1] * np.ones_like(h)

        return np.stack([h, hu, hv])


class GaussianHumpInitialConditionNoGeo(GaussianHumpInitialCondition):
    """Gaussian hump initial condition."""

    def __init__(
        self,
        height: float = 2.0,
        width: float = 0.01,
        bias: float = 0.0,
        center: tuple[float, float] = (0.0, 0.0),
        water_velocity: tuple[float, float] = (0.0, 0.0),
    ):
        """
        Create a Gaussian hump initial condition.

        Parameters
        ----------
        height : float, default=2.0
            Height of the Gaussian hump
        width : float, default=0.01
            Width parameter (controls spread)
        center : tuple[float, float], default=(0.0, 0.0)
            Center of the Gaussian hump
        """
        super(GaussianHumpInitialConditionNoGeo, self).__init__(
            height=height,
            width=width,
            bias=bias,
            center=center,
            water_velocity=water_velocity,
        )

    def get_initial_condition(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate Gaussian hump centered at domain origin.

        Returns (3, ny, nx) array with [h, hu, hv] where:
        - h: Gaussian profile
        - hu, hv: zero (no initial velocity)
        """
        # Gaussian profile
        xx, yy = np.meshgrid(
            np.linspace(0, 1, lon.shape[1]),
            np.linspace(0, 1, lat.shape[0]),
            indexing="ij",
        )  # Create a meshgrid for lon and a dummy lat
        h = self.bias + self.height * np.exp(
            -self.width * ((xx - self.center[0]) ** 2 + (yy - self.center[1]) ** 2)
        )
        hu = self.water_velocity[0] * np.ones_like(h)
        hv = self.water_velocity[1] * np.ones_like(h)

        return np.stack([h, hu, hv])


class FlatInitialCondition(InitialConditionProvider):
    """Flat water surface (rest state) initial condition."""

    def __init__(self, depth: float = 1.0):
        """
        Create a flat surface initial condition.

        Parameters
        ----------
        depth : float, default=1.0
            Water depth at rest
        """
        self.depth = depth

    def get_initial_condition(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate flat water surface.

        Returns (3, ny, nx) array with [h, hu, hv] where:
        - h: uniform depth
        - hu, hv: zero (no motion)
        """
        h = self.depth * np.ones_like(lon)
        hu = np.zeros_like(h)
        hv = np.zeros_like(h)

        return np.stack([h, hu, hv])
