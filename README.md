# SWE Simulator

A Python-based 2D Shallow Water Equations (SWE) solver using [PyClaw](http://www.clawpack.org/pyclaw/) for simulating storm surge, tsunami propagation, and coastal flooding scenarios.

## Features

- ✅ **2D Shallow Water Equations** solver with Roe-type Riemann solver
- ✅ **Geographic coordinate mapping** (lon/lat to local metric coordinates)
- ✅ **Real bathymetry support** via GEBCO NetCDF data
- ✅ **Wind forcing** for hurricane/storm surge simulations
- ✅ **MPI parallelization** for large-scale simulations
- ✅ **Adaptive time stepping** with CFL condition
- ✅ **Flexible boundary conditions** (wall, extrapolation, periodic)
- ✅ **Visualization tools** with animation support
- ✅ **Configuration management** via dataclasses and JSON

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Complete Examples](#complete-examples)
- [Output Format](#output-format)
- [Utilities](#utilities)
- [Physics](#physics)
- [Tips and Best Practices](#tips-and-best-practices)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Installation


### Install from source

```bash
git clone https://github.com/yourusername/swe_simulator.git
cd swe_simulator
pip install -r requirements.txt .
```

---

## Quick Start

### Example 1: Simple Gaussian Wave

```python
import numpy as np
import swe_simulator

# Create configuration
config = swe_simulator.SimulationConfig(
    lon_range=(-1.0, 1.0),
    lat_range=(-1.0, 1.0),
    nx=50,
    ny=50,
    t_final=10.0,
    dt=0.1,
    gravity=9.81,
    bc_lower=(0, 0),
    bc_upper=(0, 0),
    output_dir="_output",
)

# Initialize solver
solver = swe_simulator.SWESolver(config=config)

# Set flat bathymetry (-10m depth)
bathymetry = -10.0 * np.ones((config.ny, config.nx))
solver.set_bathymetry(bathymetry)

# Set Gaussian hump initial condition
x, y = solver.mapper.coord_to_metric(solver.X_coord, solver.Y_coord)
h_init = 2.0 * np.exp(-0.01 * (x**2 + y**2))
initial_condition = np.stack([
    h_init,
    np.zeros_like(h_init),  # hu (x-momentum)
    np.zeros_like(h_init),  # hv (y-momentum)
], axis=0)
solver.set_initial_condition(initial_condition)

# Run simulation
solver.setup_solver()
solutions = solver.solve()

print(f"Simulation complete! Shape: {solutions.shape}")
```

### Example 2: Storm Surge with Real Bathymetry

```python
import numpy as np
import swe_simulator
import swe_simulator.utils as sim_utils

# Configuration for Florida coast
config = swe_simulator.SimulationConfig(
    lon_range=(-80.1865, -80.0791),
    lat_range=(25.6678, 25.9137),
    nx=40,
    ny=40,
    t_final=1000.0,
    dt=1.0,
    gravity=9.81,
    bc_lower=(1, 1),
    bc_upper=(1, 1),
    output_dir="_output",
    multiple_output_times=True,
)

# Initialize solver
solver = swe_simulator.SWESolver(config=config)

# Load GEBCO bathymetry
bathymetry = sim_utils.interpolate_gebco_on_grid(
    X=solver.X_coord,
    Y=solver.Y_coord,
    nc_path="data/gebco_2025_n25.9288_s25.6527_w-80.2016_e-80.0642.nc"
)
bathymetry[np.isnan(bathymetry)] = 0.0
solver.set_bathymetry(bathymetry)

# Set radial dam break initial condition
x, y = solver.mapper.coord_to_metric(solver.X_coord, solver.Y_coord)
h_init = 0.2 + 3.0 * np.exp(-0.00001 * ((x - 3500)**2 + y**2))
initial_condition = np.stack([h_init, np.zeros_like(h_init), np.zeros_like(h_init)], axis=0)
solver.set_initial_condition(initial_condition)

# Add hurricane wind forcing (57 mph from NE)
u_wind = -17.8  # m/s
v_wind = 17.8   # m/s
solver.set_wind_forcing(u_wind=u_wind, v_wind=v_wind)

# Run simulation
solver.setup_solver()
solutions = solver.solve()

# Visualize
if solver.rank == 0:
    swe_simulator.utils.animate_solution(
        output_path=config.output_dir,
        frames=None,  # all frames
        wave_threshold=1e-2,
        interval=100,
        save=False,
    )
```

---

## Configuration

### SimulationConfig Dataclass

The `SimulationConfig` dataclass centralizes all simulation parameters:

```python
config = swe_simulator.SimulationConfig(
    # Domain
    lon_range=(-80.2, -80.0),      # Longitude range (degrees)
    lat_range=(25.6, 25.9),         # Latitude range (degrees)
    nx=40,                           # Grid cells in x
    ny=40,                           # Grid cells in y
    
    # Time
    t_final=1000.0,                  # Final time (seconds)
    dt=1.0,                          # Time step (seconds)
    
    # Physics
    gravity=9.81,                    # Gravitational acceleration (m/s²)
    
    # Boundary conditions
    bc_lower=(0, 1),     # Lower BCs [x, y]
    bc_upper=(1, 0),     # Upper BCs [x, y]
    
    # Output
    output_dir="_output",            # Output directory
    multiple_output_times=True,      # Multiple output times
    frame_interval=1,                # Frames between outputs
    
    # Numerical
    cfl_desired=0.9,                 # Desired CFL number
    cfl_max=1.0,                     # Maximum CFL number
)

# Validate configuration (automatic in __post_init__)
config.validate()

# Save configuration
config.save("config.json")

# Load configuration
config = swe_simulator.SimulationConfig.load("config.json")
```

### Boundary Conditions

Available boundary condition types:
- `'0'` - Solid wall (reflective)
- `'1'` - Extrapolation (open boundary)
- `'2' ` - Periodic boundary

```python
# Example: Open ocean boundaries
bc_lower=(1, 1)
bc_upper=(1, 1)

# Example: Coastal domain with wall on west
bc_lower=(0, 1)
bc_upper=(1, 1)
```

---

## API Reference

### SWESolver

Main solver class for running simulations.

```python
solver = swe_simulator.SWESolver(config=config)
```

#### Constructor

**`SWESolver(config: Optional[SimulationConfig] = None)`**

Initialize the shallow water equation solver.

**Parameters:**
- `config` (SimulationConfig, optional): Configuration object. If None, creates default config.

**Attributes:**
- `config` (SimulationConfig): Simulation configuration
- `X_coord`, `Y_coord` (np.ndarray): Geographic coordinate arrays (longitude, latitude)
- `mapper` (GeographicCoordinateMapper): Coordinate transformation object
- `rank` (int): MPI rank (0 for serial runs)
- `bathymetry_array` (np.ndarray): Bathymetry data
- `initial_condition_array` (np.ndarray): Initial condition data

#### Methods

**`set_bathymetry(bathymetry_array: np.ndarray) -> None`**

Set the bathymetry for the domain.

**Parameters:**
- `bathymetry_array` (np.ndarray): Array of shape `(ny, nx)` with bathymetry values in meters
  - **Negative values** represent depth below sea level (e.g., -10 means 10 meters deep)
  - **Positive values** represent elevation above sea level

**Raises:**
- `ValueError`: If bathymetry shape doesn't match grid dimensions or contains NaN/Inf

**Example:**
```python
# Flat ocean floor at 10 meters depth
bathymetry = -10.0 * np.ones((solver.config.ny, solver.config.nx))
solver.set_bathymetry(bathymetry)
```

---

**`set_initial_condition(initial_condition: np.ndarray) -> None`**

Set the initial state of the water.

**Parameters:**
- `initial_condition` (np.ndarray): 3D array of shape `(3, ny, nx)` containing:
  - `[0, :, :]`: Water depth `h` in meters
  - `[1, :, :]`: x-momentum `hu` in m²/s
  - `[2, :, :]`: y-momentum `hv` in m²/s

**Raises:**
- `ValueError`: If shape doesn't match expected dimensions, contains NaN/Inf, or has negative depths

**Example:**
```python
# Gaussian wave with zero initial momentum
x, y = solver.mapper.coord_to_metric(solver.X_coord, solver.Y_coord)
h_init = 2.0 * np.exp(-0.01 * (x**2 + y**2))
initial_condition = np.stack([
    h_init,
    np.zeros_like(h_init),  # hu
    np.zeros_like(h_init),  # hv
], axis=0)
solver.set_initial_condition(initial_condition)
```

---

**`set_wind_forcing(u_wind: float, v_wind: float, c_d: float = 1.3e-3) -> None`**

Add wind stress forcing to the simulation.

**Parameters:**
- `u_wind` (float): Wind velocity in x-direction (eastward) in m/s
- `v_wind` (float): Wind velocity in y-direction (northward) in m/s
- `c_d` (float): Drag coefficient (default: 1.3×10⁻³)

**Example:**
```python
# 10 m/s eastward wind, 5 m/s northward wind
solver.set_wind_forcing(u_wind=10.0, v_wind=5.0)

# Hurricane-force winds from northeast
speed_mph = 57
u_wind = (-1/np.sqrt(2)) * 0.44704 * speed_mph  # Convert mph to m/s
v_wind = (1/np.sqrt(2)) * 0.44704 * speed_mph
solver.set_wind_forcing(u_wind=u_wind, v_wind=v_wind)
```

---

**`setup_solver() -> None`**

Configure the PyClaw solver with all settings.

**Raises:**
- `ValueError`: If required configuration is missing

**Note:** Automatically called by `solve()` if not already called.

---

**`solve() -> np.ndarray`**

Run the simulation.

**Returns:**
- `np.ndarray`: Solution array of shape `(n_frames, 3, ny, nx)` containing water depth, x-momentum, and y-momentum at each output time.

**Raises:**
- `ValueError`: If configuration is incomplete

**Example:**
```python
solver.setup_solver()
solutions = solver.solve()
print(f"Simulation complete! Shape: {solutions.shape}")
```

---

#### Properties

**`X_coord`, `Y_coord`**
- Geographic coordinate arrays (longitude, latitude in degrees)
- Shape: `(ny, nx)`
- Available after domain initialization

**`mapper`**
- `GeographicCoordinateMapper` instance for coordinate transformations
- Methods:
  - `coord_to_metric(lon, lat)`: Convert lon/lat to local metric coordinates
  - `metric_to_coord(x, y)`: Convert metric coordinates to lon/lat

**`rank`**
- MPI rank (integer)
- 0 for serial runs or master process in parallel runs

---

## Complete Examples

### Example 1: Simple Gaussian Wave

```python
import numpy as np
import swe_simulator

# Configuration
config = swe_simulator.SimulationConfig(
    lon_range=(-10.0, 10.0),
    lat_range=(-10.0, 10.0),
    nx=100,
    ny=100,
    t_final=50.0,
    dt=0.5,
    gravity=9.81,
    bc_lower=(0, 0),
    bc_upper=(0, 0),
    output_dir="_output",
)

# Initialize solver
solver = swe_simulator.SWESolver(config=config)

# Flat bathymetry at -10m
bathymetry = -10.0 * np.ones((config.ny, config.nx))
solver.set_bathymetry(bathymetry)

# Gaussian hump initial condition
x, y = solver.mapper.coord_to_metric(solver.X_coord, solver.Y_coord)
h_init = 2.0 * np.exp(-0.0001 * (x**2 + y**2))
initial_condition = np.stack([h_init, np.zeros_like(h_init), np.zeros_like(h_init)], axis=0)
solver.set_initial_condition(initial_condition)

# Run
solver.setup_solver()
solutions = solver.solve()

print(f"Generated {solutions.shape[0]} output frames")
```

---

### Example 2: Storm Surge Simulation

```python
import numpy as np
import swe_simulator
import swe_simulator.utils as sim_utils
import functools

# Configuration for Miami area
config = swe_simulator.SimulationConfig(
    lon_range=(-80.1865, -80.0791),
    lat_range=(25.6678, 25.9137),
    nx=40,
    ny=40,
    t_final=1000.0,
    dt=1.0,
    gravity=9.81,
    bc_lower=(1, 1),
    bc_upper=(1, 1),
    output_dir="_output",
    multiple_output_times=True,
)

# Initialize solver
solver = swe_simulator.SWESolver(config=config)

# Load GEBCO bathymetry
bathymetry_interpolator = functools.partial(
    sim_utils.interpolate_gebco_on_grid,
    nc_path="data/gebco_2025_n25.9288_s25.6527_w-80.2016_e-80.0642.nc",
)
bathymetry = bathymetry_interpolator(X=solver.X_coord, Y=solver.Y_coord)
bathymetry[np.isnan(bathymetry)] = 0.0
solver.set_bathymetry(bathymetry)

# Initial condition: Radial dam break
x, y = solver.mapper.coord_to_metric(solver.X_coord, solver.Y_coord)
h_init = 0.2 + 3.0 * np.exp(-0.00001 * ((x - 3500)**2 + y**2))
initial_condition = np.stack([h_init, np.zeros_like(h_init), np.zeros_like(h_init)], axis=0)
solver.set_initial_condition(initial_condition)

# Hurricane wind forcing
speed_mph = 57
u_wind = (-1/np.sqrt(2)) * 0.44704 * speed_mph
v_wind = (1/np.sqrt(2)) * 0.44704 * speed_mph
solver.set_wind_forcing(u_wind=u_wind, v_wind=v_wind)

# Run
solver.setup_solver()
solutions = solver.solve()

print(f"Simulation complete! Shape: {solutions.shape}")
```

---

### Example 3: Visualization

```python
import swe_simulator
import swe_simulator.utils as sim_utils

# After running a simulation...

# Read all solutions
if solver.rank == 0:
    result = sim_utils.read_solutions(
        outdir=config.output_dir,
        frames_list=None,  # None = all frames
    )
    
    solutions = result["solutions"]  # (n_frames, 3, ny, nx)
    bathymetry = result["bathymetry"]
    lon_grid, lat_grid = result["meshgrid"]
    times = result["times"]
    
    print(f"Read {len(solutions)} frames")
    print(f"Time range: {times[0]:.1f} to {times[-1]:.1f} seconds")
    
    # Animate solution
    swe_simulator.utils.animate_solution(
        output_path=config.output_dir,
        frames=None,  # all frames
        wave_threshold=1e-2,
        interval=100,
        save=False,
    )
```

---

## Output Format

After running a simulation, the solver creates an output directory (default: `_output/`) containing:

### Files Generated

1. **`claw*.petsc`**: PyClaw solution files (PETSc binary format)
   - One file per output time
   - Contains state variables (depth, momentum) at that time

2. **`coord_meshgrid.npy`**: Grid coordinates
   - Shape: `(2, ny, nx)`
   - `[0, :, :]`: Longitude values
   - `[1, :, :]`: Latitude values

3. **`bathymetry.npy`**: Bathymetry data
   - Shape: `(ny, nx)`
   - Ocean floor depth/elevation values

4. **`config.json`**: Simulation configuration (if saved)
   - All configuration parameters in JSON format

### Reading Output Data

```python
import numpy as np
import swe_simulator.utils as sim_utils

# Load saved data
result = sim_utils.read_solutions(outdir="_output")

# Extract arrays
solutions = result["solutions"]    # (n_frames, 3, ny, nx)
bathymetry = result["bathymetry"]  # (ny, nx)
lon, lat = result["meshgrid"]      # (ny, nx) each
times = result["times"]            # (n_frames,)

# Access specific frame
frame_idx = 10
h = solutions[frame_idx, 0, :, :]   # Water depth
hu = solutions[frame_idx, 1, :, :]  # x-momentum
hv = solutions[frame_idx, 2, :, :]  # y-momentum

# Calculate velocities (where h > 0)
u = np.where(h > 1e-6, hu / h, 0)
v = np.where(h > 1e-6, hv / h, 0)

# Calculate free surface elevation
eta = h + bathymetry
```

---

## Utilities

### Reading Solutions

```python
from swe_simulator.utils import read_solutions

result = read_solutions(
    outdir="_output",
    frames_list=None,  # None = all frames, or list of frame numbers
    read_aux=False,    # Whether to read auxiliary variables
)

# Returns dictionary with keys:
# - 'solutions': np.ndarray (n_frames, 3, ny, nx)
# - 'bathymetry': np.ndarray (ny, nx)
# - 'meshgrid': tuple (lon_grid, lat_grid)
# - 'times': np.ndarray (n_frames,)
# - 'frames': list of frame numbers
```

### Visualization

```python
from swe_simulator.utils import animate_solution, plot_solution

# Animate all frames
animate_solution(
    output_path="_output",
    frames=None,              # None = all frames, or list of frame numbers
    wave_threshold=1e-2,      # Minimum depth to display
    interval=100,             # Milliseconds between frames
    save=False,               # Save to MP4 file
)

# Plot single frame
plot_solution(
    output_path="_output",
    frame=10,
    wave_threshold=1e-2,
)
```

### Bathymetry

```python
from swe_simulator.utils import interpolate_gebco_on_grid

# Load GEBCO bathymetry and interpolate to grid
bathymetry = interpolate_gebco_on_grid(
    X=lon_grid,
    Y=lat_grid,
    nc_path="data/gebco_file.nc"
)

# Handle NaN values (land or missing data)
bathymetry[np.isnan(bathymetry)] = 0.0
```

---

## Physics

The solver solves the 2D shallow water equations:

```
∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂y = -gh∂b/∂x + τˣ
∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + gh²/2)/∂y = -gh∂b/∂y + τʸ
```

Where:
- `h`: water depth (m)
- `u, v`: velocity components (m/s)
- `g`: gravitational acceleration (m/s²)
- `b`: bathymetry (bottom topography, m)
- `τˣ, τʸ`: wind stress terms (m²/s²)

### Wind Forcing

Wind stress is computed as:
```
τˣ = (ρₐ c_d |U| u_wind) / ρ_water
τʸ = (ρₐ c_d |U| v_wind) / ρ_water
```

Where:
- `ρₐ = 1.225 kg/m³`: air density
- `ρ_water = 1000 kg/m³`: water density
- `c_d = 1.3×10⁻³`: drag coefficient
- `U = (u_wind, v_wind)`: wind velocity (m/s)

---

## Tips and Best Practices

### 1. Grid Resolution

- **Start coarse**: Begin with `nx=40, ny=40` for testing
- **Refine gradually**: Double resolution to check convergence
- **Memory consideration**: Memory usage scales as `O(nx * ny)`
- **Typical values**: 40-200 cells per dimension

### 2. Time Step Selection

- **CFL condition**: `dt` should satisfy Courant-Friedrichs-Lewy condition
- **Rule of thumb**: `dt ≤ min(dx, dy) / sqrt(g * max_depth)`
- **Start conservative**: Use smaller `dt` initially (0.5-1.0 seconds)
- **Adaptive stepping**: Solver can adjust `dt` automatically (controlled by `cfl_desired`)

### 3. Boundary Conditions

- **Ocean boundaries**: Use `1` for open ocean boundaries
- **Coast/walls**: Use `0` for solid boundaries (coastline, islands)
- **Periodic**: Use `'2'` for periodic domains (rarely used)
- **Mixed conditions**: Different BCs on different edges

### 4. Initial Conditions

- **Positive depths**: Ensure `h ≥ 0` everywhere
- **Smooth transitions**: Avoid discontinuities in initial conditions
- **Balance**: Ensure momentum is consistent with water depth
- **Dry regions**: Set `h = 0` for initially dry land

### 5. Wind Forcing

- **Units**: Wind velocities in m/s
- **Direction convention**:
  - `u > 0`: Eastward wind
  - `v > 0`: Northward wind
- **Magnitude**: Typical values:
  - Light breeze: 5-10 m/s
  - Strong winds: 15-25 m/s
  - Hurricane: 30-70 m/s

### 6. Coordinate Systems

- **Geographic coordinates**: `X_coord`, `Y_coord` in degrees
- **Metric coordinates**: Use `mapper.coord_to_metric()` for distances
- **Array ordering**: Always `(ny, nx)` (row-major, latitude × longitude)
- **Bathymetry convention**: Negative = depth, Positive = elevation

### 7. MPI Parallelization

```bash
# Run with MPI (4 processes)
mpiexec -n 4 python your_script.py
```

- Automatically parallelized if MPI is available
- Check `solver.rank` for rank-specific operations
- Only rank 0 should do I/O and visualization
- Load balancing handled by PyClaw

### 8. Debugging

- **Validate config**: Call `config.validate()` explicitly
- **Check array shapes**: Verify `(ny, nx)` or `(3, ny, nx)` dimensions
- **Inspect bathymetry**: Plot before running simulation
- **Start simple**: Test with flat bathymetry and simple initial conditions
- **Check for NaN**: Validate bathymetry and initial condition arrays

### 9. Performance Optimization

- **Reduce output**: Set `multiple_output_times=False` for final state only
- **Increase frame_interval**: Output less frequently
- **Optimize grid**: Balance accuracy vs. computation time
- **Use MPI**: Parallelize for large domains
- **Larger dt**: Use largest stable time step

---

## Troubleshooting

### Common Errors and Solutions

**"Configuration validation failed"**
- Check that all required parameters are set
- Ensure `lon_range[0] < lon_range[1]`
- Ensure `lat_range[0] < lat_range[1]`
- Ensure `nx, ny > 0`
- Ensure `t_final > dt > 0`

**"Bathymetry shape does not match grid dimensions"**
- Ensure bathymetry shape is `(ny, nx)`, not `(nx, ny)`
- Match `config.ny` and `config.nx` exactly

**"Initial condition shape does not match expected shape"**
- Ensure shape is `(3, ny, nx)`: `[h, hu, hv]`
- First dimension must be 3 (water depth, x-momentum, y-momentum)

**"Initial condition contains NaN values"**
- Check for NaN in input arrays
- Handle NaN in bathymetry before creating initial condition

**"Initial water depth contains negative values"**
- Ensure all `h >= 0` in initial condition
- Dry areas should have `h = 0`, not negative

**ImportError: No module named 'clawpack'**
```bash
pip install clawpack
```

**MPI errors**
```bash
# Ensure mpi4py is installed correctly
pip install --upgrade mpi4py
```

**Memory issues with large grids**
- Use MPI parallelization: `mpiexec -n 4 python script.py`
- Reduce `nx`, `ny` values
- Reduce `num_output_times` or increase `frame_interval`

**Visualization not working**
```bash
# Install matplotlib and cartopy
pip install matplotlib
conda install cartopy  # Recommended method
```

**Simulation unstable (NaN or Inf in output)**
- Reduce `dt` (time step too large)
- Check initial conditions for discontinuities
- Verify bathymetry data is reasonable
- Check CFL condition: reduce `cfl_desired`

---

## Project Structure

```
swe_simulator/
├── __init__.py              # Package exports
├── config.py                # SimulationConfig dataclass
├── solver.py                # SWESolver main class
├── forcing.py               # WindForcing class
├── coordinate_mapper.py     # Geographic coordinate transformations
├── exceptions.py            # Custom exceptions
├── utils/
│   ├── __init__.py
│   ├── bathymetry.py        # Bathymetry utilities
│   ├── grid.py              # Grid generation
│   ├── io.py                # Input/output functions
│   ├── validation.py        # Validation utilities
│   └── visualization.py     # Plotting functions

examples/
├── test_sweSolver.py        # Storm surge simulation example
└── simple_gaussian.py       # Simple wave example

data/
└── gebco_*.nc               # GEBCO bathymetry NetCDF files
```

---

## Data Requirements

### GEBCO Bathymetry Data

Download bathymetry data from [GEBCO](https://www.gebco.net/):

1. Go to https://www.gebco.net/data_and_products/gridded_bathymetry_data/
2. Select your region of interest
3. Download as NetCDF format
4. Place in `data/` directory

Example filename: `gebco_2025_n25.9288_s25.6527_w-80.2016_e-80.0642.nc`

---

## References

- **Clawpack**: http://www.clawpack.org/
- **PyClaw Documentation**: http://www.clawpack.org/pyclaw/
- **Shallow Water Equations**: Classical 2D SWE with bathymetry
- **Riemann Solver**: Uses `shallow_roe_with_efix_2D`
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*
- GEBCO Bathymetric Data: https://www.gebco.net/

---

## License

MIT License - see LICENSE file for details.

---

