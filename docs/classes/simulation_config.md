# SimulationConfig

`SimulationConfig` is the central configuration dataclass used by `SWESolver`.

## Initialization Arguments

### Domain

- `lon_range: tuple[float, float] = (0.0, 1.0)`
- `lat_range: tuple[float, float] = (0.0, 1.0)`
- `nx: int = 100`
- `ny: int = 100`

### Time

- `t_final: float = 10.0`
- `dt: float = 0.1`

### Physics

- `gravity: float = 9.81`

### Output

- `output_dir: str = "_output"`
- `frame_interval: int = 1`
- `multiple_output_times: bool = False`

### Numerical

- `cfl_desired: float = 0.9`
- `cfl_max: float = 1.0`

### Boundary Conditions

- `bc_lower: tuple[pyclaw.BC, pyclaw.BC] = (pyclaw.BC.wall, pyclaw.BC.wall)`
- `bc_upper: tuple[pyclaw.BC, pyclaw.BC] = (pyclaw.BC.wall, pyclaw.BC.wall)`

## Attributes

All initialization arguments are stored as dataclass attributes and validated in `__post_init__()`.

Important behavior:

- `validate()` is called automatically at construction.
- boundary conditions accept only `pyclaw.BC.wall`, `pyclaw.BC.extrap`, `pyclaw.BC.periodic`.
- JSON load normalizes tuple/int/bool types before constructing the class.

## Methods

### `validate() -> None`

Checks domain ordering, grid shape, positive physics/time parameters, CFL constraints, boundary condition values, and output flags.

### `to_dict() -> dict[str, Any]`

Returns a plain dictionary representation of the dataclass.

### `save(filepath: str | Path) -> None`

Writes the configuration as JSON.

### `load(filepath: str | Path) -> SimulationConfig`

Class method that reads JSON and returns a validated `SimulationConfig`.

### `__str__() -> str`

Human-readable summary with domain, grid, time, physics, output, CFL, and BC values.

## Example

```python
from tidalflow.config import SimulationConfig
import clawpack.petclaw as pyclaw

config = SimulationConfig(
    lon_range=(-80.1865, -80.0791),
    lat_range=(25.6678, 25.9137),
    nx=80,
    ny=80,
    t_final=1200.0,
    dt=2.0,
    gravity=9.81,
    bc_lower=(pyclaw.BC.extrap, pyclaw.BC.extrap),
    bc_upper=(pyclaw.BC.extrap, pyclaw.BC.extrap),
    output_dir="_output",
    multiple_output_times=True,
)

config.save("config.json")
loaded = SimulationConfig.load("config.json")
```
