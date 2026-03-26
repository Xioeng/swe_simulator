"""Wind forcing and source terms for shallow water equations."""

import numpy as np
import numpy.typing as npt
from clawpack.riemann.shallow_roe_with_efix_2D_constants import (
    depth,
    x_momentum,
    y_momentum,
)

from .logging_config import get_logger
from .providers.base import WindProvider
from .providers.wind import ConstantWind

logger = get_logger(__name__)


class WindForcing:
    """
    Wind forcing source term for shallow water equations.

    Parameters
    ----------
    u_wind : float, default=0.0
        Wind velocity in x-direction (m/s)
    v_wind : float, default=0.0
        Wind velocity in y-direction (m/s)
    c_d : float, default=1.3e-3
        Air-water drag coefficient (dimensionless)
    rho_air : float, default=1.2
        Air density (kg/m³)
    rho_water : float, default=1000.0
        Water density (kg/m³)
    """

    def __init__(
        self,
        mesgrid_domain: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        c_d: float = 1.3e-3,
        rho_air: float = 1.2,
        rho_water: float = 1000.0,
        wind_provider: WindProvider = ConstantWind(),
    ) -> None:
        self.c_d = c_d
        self.rho_air = rho_air
        self.rho_water = rho_water
        self.X_coord, self.Y_coord = mesgrid_domain
        self.wind_provider = wind_provider

        logger.debug(
            f"WindForcing initialized: u={np.mean(self.get_wind()[0]):.2f} m/s,"
            f"v={np.mean(self.get_wind()[1]):.2f} m/s, "
            f"C_D={c_d:.2e}"
        )

    def get_wind(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get current wind velocity components.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            Current wind velocities in m/s
        """
        return self.wind_provider.get_wind(self.X_coord, self.Y_coord, time=0)

    def set_drag_coefficient(self, c_d: float) -> None:
        """
        Set the air-water drag coefficient.

        Parameters
        ----------
        c_d : float
            Drag coefficient (dimensionless, typically 1e-3 to 2e-3)
        """
        self.c_d = c_d
        logger.info(f"Drag coefficient set to: {c_d:.2e}")

    def get_drag_coefficient(self) -> float:
        """
        Get current drag coefficient.

        Returns
        -------
        float
            Current drag coefficient
        """
        return self.c_d

    @staticmethod
    def compute_velocities(
        h: np.ndarray, hu: np.ndarray, hv: np.ndarray, threshold: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute water velocities from momentum and depth arrays.

        Parameters
        ----------
        h : np.ndarray
            Water depth (m)
        hu : np.ndarray
            X-momentum (m³/s)
        hv : np.ndarray
            Y-momentum (m³/s)
        threshold : float, default=1e-6
            Minimum water depth threshold for velocity calculation

        Returns
        -------
        u : np.ndarray
            X-velocity (m/s)
        v : np.ndarray
            Y-velocity (m/s)
        """
        u = np.zeros_like(h)
        v = np.zeros_like(h)

        # Avoid division by zero for dry cells
        mask = h > threshold
        u[mask] = hu[mask] / h[mask]
        v[mask] = hv[mask] / h[mask]

        return u, v, mask

    def compute_wind_stress(
        self,
        h: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate wind stress on water surface.

        Parameters
        ----------
        h : np.ndarray
            Water depth (m)
        u : np.ndarray
            X-velocity (m/s)
        v : np.ndarray
            Y-velocity (m/s)

        Returns
        -------
        tau_x : np.ndarray
            Wind stress in x-direction (m²/s²)
        tau_y : np.ndarray
            Wind stress in y-direction (m²/s²)
        """
        # Relative wind velocity

        u_rel = u_wind - u
        v_rel = v_wind - v
        wind_speed_rel = np.sqrt(u_rel**2 + v_rel**2)

        # Wind stress: τ = (ρ_air/p_water) * C_D * |U_wind - U_water| * (U_wind - U_water)
        tau_x = (self.rho_air / self.rho_water) * self.c_d * wind_speed_rel * u_rel
        tau_y = (self.rho_air / self.rho_water) * self.c_d * wind_speed_rel * v_rel

        return tau_x, tau_y

    def __call__(self, solver, state, dt):
        """
        Source term function for PyClaw solver (vectorized wind forcing).

        This makes the class instance callable, compatible with PyClaw's
        step_source interface.

        Parameters
        ----------
        solver : pyclaw.Solver
            PyClaw solver instance
        state : pyclaw.State
            Current state
        dt : float
            Time step size

        Returns
        -------
        float
            Always returns dt (no time step modification)
        """
        q = state.q

        h = q[depth, :, :]
        hu = q[x_momentum, :, :]
        hv = q[y_momentum, :, :]

        u_wind, v_wind = self.wind_provider.get_wind(
            self.X_coord, self.Y_coord, time=state.t
        )

        # Calculate water velocities
        u, v, mask = self.compute_velocities(h, hu, hv, threshold=1e-6)

        tau_x, tau_y = self.compute_wind_stress(h, u, v, u_wind, v_wind)
        # Update momentum where there's water
        q[x_momentum, :, :] += dt * h * tau_x * mask
        q[y_momentum, :, :] += dt * h * tau_y * mask

        return dt
