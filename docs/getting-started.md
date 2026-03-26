# Getting Started

This page shows the minimum workflow to run a simulation.

## 1) Create a config

```python
from tidalflow.config import SimulationConfig

config = SimulationConfig(
    lon_range=(-1.0, 1.0),
    lat_range=(-1.0, 1.0),
    nx=50,
    ny=50,
    t_final=10.0,
    dt=0.1,
)
```

## 2) Create a solver

```python
from tidalflow.solver import SWESolver

solver = SWESolver(config=config)
```

## 3) Set bathymetry and initial condition

```python
import numpy as np

bathymetry = -10.0 * np.ones((config.ny, config.nx))
solver.set_bathymetry(bathymetry)

h = np.ones((config.ny, config.nx))
hu = np.zeros_like(h)
hv = np.zeros_like(h)
solver.set_initial_condition(np.stack([h, hu, hv], axis=0))
```

## 4) Run

```python
solver.setup_solver()
result = solver.solve()
```

See class-level details in [SWESolver](classes/swe_solver.md).
