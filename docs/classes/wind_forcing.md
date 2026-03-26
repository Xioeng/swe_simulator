# WindForcing

`WindForcing` implements wind source terms for SWE momentum equations and is attached to PyClaw via `solver.step_source`.

## Initialization Arguments

- `mesgrid_domain: tuple[np.ndarray, np.ndarray]`
  - Geographic meshgrid `(X_coord, Y_coord)` used to query wind.
- `c_d: float = 1.3e-3`
  - Air-water drag coefficient.
- `rho_air: float = 1.2`
  - Air density in kg/m³.
- `rho_water: float = 1000.0`
  - Water density in kg/m³.
- `wind_provider: WindProvider = ConstantWind()`
  - Provider that returns `(u_wind, v_wind)` as arrays.

## Attributes

- `c_d`, `rho_air`, `rho_water`
- `X_coord`, `Y_coord`
- `wind_provider`

## Methods

### `get_wind() -> tuple[np.ndarray, np.ndarray]`

Returns wind components from the configured provider at `time=0`.

### `set_drag_coefficient(c_d: float) -> None`

Updates drag coefficient.

### `get_drag_coefficient() -> float`

Returns current drag coefficient.

### `compute_velocities(h, hu, hv, threshold=1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]`

Static helper that computes water velocities and wet-cell mask from state variables.

### `compute_wind_stress(h, u, v, u_wind, v_wind) -> tuple[np.ndarray, np.ndarray]`

Computes wind stress terms:

- `tau_x = (rho_air / rho_water) * c_d * |U_rel| * u_rel`
- `tau_y = (rho_air / rho_water) * c_d * |U_rel| * v_rel`

where $U_{rel} = U_{wind} - U_{water}$.

### `__call__(solver, state, dt) -> float`

PyClaw source-step callback that applies momentum updates and returns `dt`.

## Usage

`SWESolver.setup_solver()` wires this class automatically when wind forcing is available.

```python
from tidalflow.forcing import WindForcing
from tidalflow.providers import ConstantWind

forcing = WindForcing(
    mesgrid_domain=(X_coord, Y_coord),
    wind_provider=ConstantWind(u_wind=5.0, v_wind=2.0),
)
```
