# Providers

Providers are the data interfaces used by `SWESolver` to populate initial conditions, bathymetry, and wind fields.

## Base Interfaces

### `InitialConditionProvider` (abstract)

- **Method**: `get_initial_condition(lon, lat) -> np.ndarray`
- **Expected return**: array with shape `(3, ny, nx)` containing `[h, hu, hv]`.

### `BathymetryProvider` (abstract)

- **Method**: `get_bathymetry(lon, lat) -> np.ndarray`
- **Expected return**: array with shape `(ny, nx)`.

### `WindProvider` (abstract)

- **Method**: `get_wind(lon, lat, time) -> tuple[np.ndarray, np.ndarray]`
- **Expected return**: `(u_wind, v_wind)` arrays, each shape `(ny, nx)`.

## Initial Condition Providers

### `GaussianHumpInitialCondition`

#### Initialization Arguments

- `height: float = 2.0`
- `width: float = 0.01`
- `bias: float = 0.0`
- `center: tuple[float, float] = (0.0, 0.0)`
- `water_velocity: tuple[float, float] = (0.0, 0.0)`

#### Attributes

- `height`, `width`, `bias`, `center`, `water_velocity`

#### Methods

- `get_initial_condition(lon, lat) -> np.ndarray`

### `GaussianHumpInitialConditionNoGeo`

Same constructor signature as `GaussianHumpInitialCondition`, but evaluates the Gaussian on a normalized synthetic mesh instead of geographic coordinates.

#### Methods

- `get_initial_condition(lon, lat) -> np.ndarray`

### `FlatInitialCondition`

#### Initialization Arguments

- `depth: float = 1.0`

#### Attributes

- `depth`

#### Methods

- `get_initial_condition(lon, lat) -> np.ndarray`

## Bathymetry Providers

### `FlatBathymetry`

#### Initialization Arguments

- `depth: float = -10.0`

#### Attributes

- `depth`

#### Methods

- `get_bathymetry(lon, lat) -> np.ndarray`

### `SlopingBathymetry`

#### Initialization Arguments

- `depth_min: float = -5.0`
- `depth_max: float = -20.0`

#### Attributes

- `depth_min`, `depth_max`

#### Methods

- `get_bathymetry(lon, lat) -> np.ndarray`

### `BathymetryFromNC`

#### Initialization Arguments

- `nc_path: str | Path`

#### Attributes

- `nc_path`
- `bathymetry_interpolator` (partial function around `utils.bathymetry.interpolate_gebco_on_grid`)

#### Methods

- `get_bathymetry(lon, lat) -> np.ndarray`

## Wind Providers

### `ConstantWind`

#### Initialization Arguments

- `u_wind: float = 0.0`
- `v_wind: float = 0.0`

#### Attributes

- `u_wind`, `v_wind`

#### Methods

- `get_wind(lon, lat, time) -> tuple[np.ndarray, np.ndarray]`

## Example: Wiring Providers into SWESolver

```python
from tidalflow.config import SimulationConfig
from tidalflow.providers import (
    BathymetryFromNC,
    ConstantWind,
    GaussianHumpInitialCondition,
)
from tidalflow.solver import SWESolver

config = SimulationConfig(nx=40, ny=40, t_final=1000.0, dt=1.0)

solver = SWESolver(
    config=config,
    ic_provider=GaussianHumpInitialCondition(height=2.0, width=0.01),
    bathymetry_provider=BathymetryFromNC("data/your_gebco_file.nc"),
    wind_provider=ConstantWind(u_wind=5.0, v_wind=2.0),
)

solver.setup_solver()
result = solver.solve()
```
