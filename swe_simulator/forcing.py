"""Wind forcing and source terms for shallow water equations."""

import logging

import numpy as np
from clawpack.riemann.shallow_roe_with_efix_2D_constants import (
    depth,
    x_momentum,
    y_momentum,
)

logger = logging.getLogger(__name__)


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
        u_wind: float = 0.0,
        v_wind: float = 0.0,
        c_d: float = 1.3e-3,
        rho_air: float = 1.2,
        rho_water: float = 1000.0,
    ):
        self.u_wind = u_wind
        self.v_wind = v_wind
        self.c_d = c_d
        self.rho_air = rho_air
        self.rho_water = rho_water

        logger.debug(
            f"WindForcing initialized: u={u_wind:.2f} m/s, v={v_wind:.2f} m/s, "
            f"C_D={c_d:.2e}"
        )

    def set_wind(self, u_wind: float, v_wind: float) -> None:
        """
        Set wind velocity components.

        Parameters
        ----------
        u_wind : float
            Wind velocity in x-direction (m/s)
        v_wind : float
            Wind velocity in y-direction (m/s)
        """
        self.u_wind = u_wind
        self.v_wind = v_wind
        logger.info(f"Wind set to: u={u_wind:.2f} m/s, v={v_wind:.2f} m/s")

    def get_wind(self) -> tuple[float, float]:
        """
        Get current wind velocity components.

        Returns
        -------
        u_wind : float
            Wind velocity in x-direction (m/s)
        v_wind : float
            Wind velocity in y-direction (m/s)
        """
        return self.u_wind, self.v_wind

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

    def wind_stress(
        self,
        h: float,
        hu: float,
        hv: float,
    ) -> tuple[float, float]:
        """
        Calculate wind stress on water surface.

        Parameters
        ----------
        h : float
            Water depth (m)
        hu : float
            X-momentum (m²/s)
        hv : float
            Y-momentum (m²/s)

        Returns
        -------
        tau_x : float
            Wind stress in x-direction (m²/s²)
        tau_y : float
            Wind stress in y-direction (m²/s²)
        """
        if h < 1e-6:
            return 0.0, 0.0

        # Water velocities
        u = hu / h
        v = hv / h

        # Relative wind velocity
        u_rel = self.u_wind - u
        v_rel = self.v_wind - v
        wind_speed_rel = np.sqrt(u_rel**2 + v_rel**2)

        # Wind stress: τ = ρ_air * C_D * |U_wind - U_water| * (U_wind - U_water)
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

        # Only apply where water depth is significant
        mask = h > 1e-6

        # Calculate water velocities
        u = np.zeros_like(h)
        v = np.zeros_like(h)
        u[mask] = hu[mask] / h[mask]
        v[mask] = hv[mask] / h[mask]

        # Relative wind velocity
        u_rel = self.u_wind - u
        v_rel = self.v_wind - v
        wind_speed_rel = np.sqrt(u_rel**2 + v_rel**2)

        # Wind stress
        tau_x = (self.rho_air / self.rho_water) * self.c_d * wind_speed_rel * u_rel
        tau_y = (self.rho_air / self.rho_water) * self.c_d * wind_speed_rel * v_rel

        # Update momentum where there's water
        q[x_momentum, :, :] += dt * h * tau_x * mask
        q[y_momentum, :, :] += dt * h * tau_y * mask

        return dt
