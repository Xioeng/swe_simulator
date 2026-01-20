import matplotlib.pyplot as plt
import numpy as np
from clawpack import riemann
from clawpack.riemann.shallow_roe_with_efix_2D_constants import (
    depth,
    num_eqn,
    x_momentum,
    y_momentum,
)

# At top of swe2D_bath.py, you can define air density, water density, Cd, and wind:
rho_air = 1.2  # kg/m^3
rho_water = 1000  # kg/m^3
C_D = 1.3e-3  # dimensionless air-water drag coefficient

# Example: uniform wind in +x direction (10 m/s)
speed_florida = 57  # mph
U_a = (-1 / np.sqrt(2)) * 0.44 * speed_florida  # m/s
V_a = (1 / np.sqrt(2)) * 0.44 * speed_florida  # m/s


def set_wind(U_wind: float, V_wind: float) -> None:
    """
    Sets the uniform wind velocity components.

    Args:
        U_wind (float): Wind velocity in the x-direction (m/s).
        V_wind (float): Wind velocity in the y-direction (m/s).
    """
    global U_a, V_a
    U_a = U_wind
    V_a = V_wind


def wind_forcing_step(solver: object, state: object, dt: float) -> None:
    """
    Applies wind forcing to the shallow water equations.

    Args:
        solver (object): The solver object (not used in this function).
        state (object): The state object containing the solution array `q`.
        dt (float): The time step size.

    The function modifies the momentum components of the solution array `q`
    in-place to account for wind forcing.
    """
    q = state.q
    h = q[depth, :, :]
    hu = q[x_momentum, :, :]
    hv = q[y_momentum, :, :]

    # Water velocities
    u = np.where(h > 1e-6, hu / h, 0.0)
    v = np.where(h > 1e-6, hv / h, 0.0)

    # Relative wind (air speed minus water speed)
    U_rel = U_a - u
    V_rel = V_a - v
    speed_rel = np.sqrt(U_rel**2 + V_rel**2)

    # Wind stress (vector) on water surface
    tau_x = rho_air * C_D * speed_rel * U_rel
    tau_y = rho_air * C_D * speed_rel * V_rel

    # Body force (acceleration) on water column
    # a = tau / (rho_water * h)
    # guard against very small h:
    h_eff = np.where(h > 1e-8, h, 1e-8)
    ax = tau_x / (rho_water * h_eff)
    ay = tau_y / (rho_water * h_eff)

    # Update momentum: (hu)_t += h * ax, (hv)_t += h * ay
    hu += h * ax * dt
    hv += h * ay * dt

    q[x_momentum, :, :] = hu
    q[y_momentum, :, :] = hv
