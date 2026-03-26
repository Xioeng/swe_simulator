# SWESolver

`SWESolver` orchestrates domain setup, data initialization, PyClaw configuration, and time integration for 2D shallow water equations.

## Initialization Arguments

- `config: SimulationConfig | None = None`
  - Main simulation configuration. If `None`, defaults are used.
- `ic_provider: InitialConditionProvider | None = None`
  - Optional provider to generate the initial condition array.
- `wind_provider: WindProvider = ConstantWind()`
  - Wind provider used by `WindForcing`.
- `bathymetry_provider: BathymetryProvider | None = None`
  - Optional provider to generate bathymetry.

## Attributes

### Core Configuration

- `config: SimulationConfig`
- `ic_provider: InitialConditionProvider | None`
- `wind_provider: WindProvider`
- `bathymetry_provider: BathymetryProvider | None`

### Domain/Coordinates (after `set_domain`)

- `mapper: GeographicCoordinateMapper`
- `x_domain: list[float]`
- `y_domain: list[float]`
- `X, Y`: metric-space meshgrid cell centers
- `X_coord, Y_coord`: geographic meshgrid cell centers

### State Arrays

- `bathymetry_array: np.ndarray` with shape `(ny, nx)`
- `initial_condition_array: np.ndarray` with shape `(3, ny, nx)`
- `wind_forcing: WindForcing` (created during provider initialization)

### Runtime/MPI

- `comm: MPI.Comm`
- `rank: int`
- `claw: pyclaw.Controller` (after `setup_solver`)

## Methods

### Configuration and Domain

- `set_time_parameters(t_final: float, dt: float) -> None`
- `set_domain(lon_range: tuple[float, float], lat_range: tuple[float, float], nx: int, ny: int) -> None`
- `set_boundary_conditions(lower: tuple[int, int], upper: tuple[int, int]) -> None`

### Data Setup

- `initialize_data_from_providers() -> None`
  - Populates bathymetry and initial condition from providers and creates `WindForcing`.
- `set_bathymetry(bathymetry_array: np.ndarray) -> None`
  - Manually sets bathymetry and disables `bathymetry_provider`.
- `set_initial_condition(initial_condition: np.ndarray) -> None`
  - Manually sets initial condition and disables `ic_provider`.
- `set_constant_wind_forcing(u_wind: float = 0.0, v_wind: float = 0.0) -> None`
  - Replaces current wind provider with `ConstantWind`.

### Solve Workflow

- `setup_solver() -> pyclaw.Controller`
  - Builds solver/domain/state/controller and applies source terms.
- `solve() -> SWEResult`
  - Runs the simulation and returns `SWEResult`.
  - Writes `result.pkl` to `config.output_dir` on rank 0.

## Typical Usage

```python
import numpy as np
from tidalflow.config import SimulationConfig
from tidalflow.solver import SWESolver

config = SimulationConfig(nx=50, ny=50, t_final=100.0, dt=1.0)
solver = SWESolver(config=config)

bathymetry = -10.0 * np.ones((config.ny, config.nx))
solver.set_bathymetry(bathymetry)

h = np.ones((config.ny, config.nx))
hu = np.zeros_like(h)
hv = np.zeros_like(h)
solver.set_initial_condition(np.stack([h, hu, hv], axis=0))

solver.setup_solver()
result = solver.solve()
```
