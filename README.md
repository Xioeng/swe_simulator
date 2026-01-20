# sweSolver Class Documentation

## Overview

`sweSolver` is a Python class for solving 2D shallow water equations (SWE) using the PyClaw library from Clawpack. It provides a high-level interface for simulating water flow over bathymetry with support for geographic coordinates (longitude/latitude), wind forcing, and customizable boundary conditions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Complete Examples](#complete-examples)
- [Output Format](#output-format)
- [Tips and Best Practices](#tips-and-best-practices)

---

## Features

- **Geographic Coordinate System**: Works with longitude/latitude coordinates and automatically converts to metric space
- **Flexible Bathymetry**: Support for complex ocean floor topography
- **Initial Conditions**: Set water surface elevation and momentum fields
- **Wind Forcing**: Add atmospheric wind forcing to the simulation
- **Boundary Conditions**: Multiple types (wall, extrapolation, periodic)
- **MPI Parallelization**: Built-in support for parallel computing
- **Automatic Output**: Saves solutions, grid coordinates, and bathymetry data

---

## Installation

### Required Dependencies

```bash
pip install clawpack mpi4py numpy
```

### Optional Dependencies (for visualization)

```bash
pip install matplotlib
```

---

## Quick Start

```python
import numpy as np
import clawpack.petclaw as pyclaw
from sweSolver import sweSolver

# 1. Initialize solver
solver = sweSolver(multiple_output_times=True)

# 2. Configure domain
solver.set_domain(
    lon_range=(-80.20, -80.06),  # Longitude range (degrees)
    lat_range=(25.65, 25.93),    # Latitude range (degrees)
    nx=40,                        # Grid cells in x
    ny=40                         # Grid cells in y
)

# 3. Set time parameters
solver.set_time_parameters(t_final=100.0, dt=1.0)

# 4. Define bathymetry (ocean floor depth, positive values)
bathymetry = 10 * np.ones((solver.ny, solver.nx))
solver.set_bathymetry(bathymetry_array=bathymetry)

# 5. Define initial conditions [surface elevation, x-momentum, y-momentum]
X_metric, Y_metric = solver.mapper.coord_to_metric(solver.X_coord, solver.Y_coord)
eta = 0.2 + 3 * np.exp(-0.00001 * ((X_metric - 3500)**2 + Y_metric**2))
initial_condition = np.stack([eta, np.zeros_like(eta), np.zeros_like(eta)], axis=0)
solver.set_initial_condition(initial_condition=initial_condition)

# 6. Run simulation
solver.solve()
```

---

## API Reference

### Constructor

#### `sweSolver(multiple_output_times=True)`

Initialize the shallow water equation solver.

**Parameters:**
- `multiple_output_times` (bool, default=True): 
  - If `True`: Outputs solution at every time step
  - If `False`: Only outputs the final solution

**Attributes:**
- `gravity` (float): Gravitational acceleration (default: 9.81 m/s²)
- `nx`, `ny` (int): Number of grid cells in x and y directions
- `X_coord`, `Y_coord` (np.ndarray): Grid coordinates in lon/lat
- `mapper` (LocalLonLatMetricMapper): Coordinate transformation object
- `claw` (pyclaw.Controller): PyClaw controller after `setup_solver()` is called

---

### Methods

#### `set_domain(lon_range, lat_range, nx, ny)`

Set up the computational domain in geographic coordinates.

**Parameters:**
- `lon_range` (tuple): `(min_longitude, max_longitude)` in degrees
- `lat_range` (tuple): `(min_latitude, max_latitude)` in degrees
- `nx` (int): Number of grid cells in x-direction (longitude)
- `ny` (int): Number of grid cells in y-direction (latitude)

**Effects:**
- Creates coordinate mapper for lon/lat to metric conversion
- Generates cell-centered grid coordinates
- Stores grid dimensions

**Example:**
```python
solver.set_domain(
    lon_range=(-80.2015, -80.0641),
    lat_range=(25.6528, 25.9287),
    nx=40,
    ny=40
)
```

---

#### `set_time_parameters(t_final, dt)`

Configure the simulation time settings.

**Parameters:**
- `t_final` (float): Final simulation time in seconds
- `dt` (float): Time step size in seconds

**Note:** The number of output times is calculated as `int(t_final / dt)` when `multiple_output_times=True`.

**Example:**
```python
solver.set_time_parameters(t_final=100.0, dt=1.0)  # 100 seconds, 1-second steps
```

---

#### `set_bathymetry(bathymetry_array)`

Define the ocean floor topography.

**Parameters:**
- `bathymetry_array` (np.ndarray): 2D array of shape `(ny, nx)` containing depth values in meters
  - **Negative values** represent depth below sea level (e.g., -10 means 10 meters deep)
  - **Positive values** represent elevation above sea level
  - Must match the grid dimensions set in `set_domain()`

**Example:**
```python
# Flat ocean floor at 10 meters depth
bathymetry = -10 * np.ones((solver.ny, solver.nx))
solver.set_bathymetry(bathymetry_array=bathymetry)

# Sloping bathymetry
x = np.linspace(0, 1, solver.nx)
y = np.linspace(0, 1, solver.ny)
X, Y = np.meshgrid(x, y)
bathymetry = -20 + 15 * X  # Slopes from -20m to -5m
solver.set_bathymetry(bathymetry_array=bathymetry)
```

---

#### `set_initial_condition(initial_condition)`

Set the initial state of the water.

**Parameters:**
- `initial_condition` (np.ndarray): 3D array of shape `(3, ny, nx)` containing:
  - `[0, :, :]`: Water surface elevation η (eta) in meters above/below mean sea level
  - `[1, :, :]`: x-momentum (hu) in m²/s
  - `[2, :, :]`: y-momentum (hv) in m²/s

**Note:** The actual water depth `h` is computed as `h = max(0, η - bathymetry)`.

**Example:**
```python
# Gaussian wave with zero initial momentum
X_metric, Y_metric = solver.mapper.coord_to_metric(solver.X_coord, solver.Y_coord)
eta = 0.2 + 3 * np.exp(-0.00001 * ((X_metric - 3500)**2 + Y_metric**2))
momentum_x = np.zeros_like(eta)
momentum_y = np.zeros_like(eta)
initial_condition = np.stack([eta, momentum_x, momentum_y], axis=0)
solver.set_initial_condition(initial_condition=initial_condition)
```

---

#### `set_boundary_conditions(lower, upper)`

Configure boundary conditions for the domain edges.

**Parameters:**
- `lower` (list): `[x_lower_BC, y_lower_BC]` - Boundary conditions for lower edges
- `upper` (list): `[x_upper_BC, y_upper_BC]` - Boundary conditions for upper edges

**Available Boundary Condition Types:**
- `pyclaw.BC.wall` (0): Reflective wall (zero normal velocity)
- `pyclaw.BC.extrap` (1): Extrapolation (non-reflective outflow)
- `pyclaw.BC.periodic` (2): Periodic boundary

**Default:** All boundaries are set to `pyclaw.BC.wall`.

**Example:**
```python
# Wall on x-lower and y-upper, extrapolation on others
solver.set_boundary_conditions(
    lower=[pyclaw.BC.wall, pyclaw.BC.extrap],
    upper=[pyclaw.BC.extrap, pyclaw.BC.wall]
)
```

---

#### `set_forcing(wind_vector)`

Add wind forcing to the simulation.

**Parameters:**
- `wind_vector` (tuple): `(U_wind, V_wind)` - Wind velocity components in m/s
  - `U_wind`: Wind velocity in x-direction (eastward)
  - `V_wind`: Wind velocity in y-direction (northward)

**Example:**
```python
# 10 m/s eastward wind, 5 m/s northward wind
solver.set_forcing(wind_vector=(10.0, 5.0))

# Hurricane-force winds (southwest to northeast)
speed_mph = 57
U_wind = (-1/np.sqrt(2)) * 0.44 * speed_mph  # Convert mph to m/s
V_wind = (1/np.sqrt(2)) * 0.44 * speed_mph
solver.set_forcing(wind_vector=(U_wind, V_wind))
```

---

#### `setup_solver()`

Construct the PyClaw solver, domain, and controller.

**Returns:** `pyclaw.Controller` object

**Note:** This method is automatically called by `solve()` if not already called. It validates that all required configuration has been set.

**Raises:**
- `RuntimeError`: If domain, initial conditions, bathymetry, or time parameters are not set

---

#### `solve()`

Run the simulation.

**Returns:** Status code from PyClaw controller

**Note:** Automatically calls `setup_solver()` if not already called.

**Example:**
```python
status = solver.solve()
print(f"Simulation completed with status: {status}")
```

---

## Complete Examples

### Example 1: Simple Gaussian Wave

```python
import numpy as np
import clawpack.petclaw as pyclaw
from sweSolver import sweSolver

# Initialize
solver = sweSolver(multiple_output_times=True)

# Domain: Small region
solver.set_domain(
    lon_range=(-10.0, 10.0),
    lat_range=(-10.0, 10.0),
    nx=100,
    ny=100
)

# Time: 50 seconds
solver.set_time_parameters(t_final=50.0, dt=0.5)

# Bathymetry: Flat at -10m
bathymetry = -10 * np.ones((solver.ny, solver.nx))
solver.set_bathymetry(bathymetry_array=bathymetry)

# Initial condition: Gaussian hump
X_metric, Y_metric = solver.mapper.coord_to_metric(solver.X_coord, solver.Y_coord)
eta = 2.0 * np.exp(-0.0001 * (X_metric**2 + Y_metric**2))
initial_condition = np.stack([eta, np.zeros_like(eta), np.zeros_like(eta)], axis=0)
solver.set_initial_condition(initial_condition=initial_condition)

# Boundary: All walls
solver.set_boundary_conditions(
    lower=[pyclaw.BC.wall, pyclaw.BC.wall],
    upper=[pyclaw.BC.wall, pyclaw.BC.wall]
)

# Run
solver.solve()
```

---

### Example 2: Storm Surge Simulation with Real Bathymetry

```python
import numpy as np
import clawpack.petclaw as pyclaw
from sweSolver import sweSolver
import utils

# Initialize
solver = sweSolver(multiple_output_times=True)

# Domain: Miami area
lon_range = (-80.2015, -80.0641)
lat_range = (25.6528, 25.9287)
solver.set_domain(lon_range=lon_range, lat_range=lat_range, nx=40, ny=40)

# Time: 100 seconds with 1s steps
solver.set_time_parameters(t_final=100.0, dt=1.0)

# Load real bathymetry from GEBCO data
bathymetry_interpolator = utils.interpolate_gebco_on_grid(
    nc_path="data/gebco_2025_n25.9288_s25.6527_w-80.2016_e-80.0642.nc"
)
bathymetry = bathymetry_interpolator(X=solver.X_coord, Y=solver.Y_coord)
bathymetry[np.isnan(bathymetry)] = 0.0
solver.set_bathymetry(bathymetry_array=bathymetry)

# Initial condition: Tide with localized surge
X_metric, Y_metric = solver.mapper.coord_to_metric(solver.X_coord, solver.Y_coord)
eta = 0.2 + 3 * np.exp(-0.00001 * ((X_metric - 3500)**2 + Y_metric**2))
initial_condition = np.stack([eta, np.zeros_like(eta), np.zeros_like(eta)], axis=0)
solver.set_initial_condition(initial_condition=initial_condition)

# Boundary: Mixed conditions
solver.set_boundary_conditions(
    lower=[pyclaw.BC.wall, pyclaw.BC.extrap],
    upper=[pyclaw.BC.extrap, pyclaw.BC.wall]
)

# Wind forcing: Hurricane conditions
speed_mph = 57
U_wind = (-1/np.sqrt(2)) * 0.44 * speed_mph
V_wind = (1/np.sqrt(2)) * 0.44 * speed_mph
solver.set_forcing(wind_vector=(U_wind, V_wind))

# Run
solver.solve()
```

---

### Example 3: Visualization of Results

```python
import numpy as np
import matplotlib.pyplot as plt
from sweSolver import sweSolver
import utils

# ... (setup and run solver as in previous examples)

# Read solutions
result = utils.read_solutions(
    outdir="_output",
    frames_list=list(range(len(solver.claw.frames)))
)

solutions = result["solutions"][:, 0, ...]  # Extract depth component
X_coord, Y_coord = result["meshgrid"]
bath = result["bathymetry"]

# Create animation
fig, ax = plt.subplots(figsize=(10, 8))

for i, h in enumerate(solutions):
    ax.clear()
    
    # Calculate free surface elevation
    free_surface = h + bath
    free_surface[h < 1e-3] = np.nan  # Mask dry cells
    
    im = ax.pcolormesh(X_coord, Y_coord, free_surface, shading='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Water Surface at t = {i * solver.dt:.1f}s')
    ax.set_aspect('equal')
    plt.pause(0.1)

plt.show()
```

---

## Output Format

After running a simulation, the solver creates an `_output/` directory containing:

### Files Generated

1. **`claw.pkl####`**: PyClaw solution files (pickle format)
   - One file per output time
   - Contains state variables (depth, momentum) at that time
   - Can be loaded with PyClaw's `Solution` class

2. **`coord_meshgrid.npy`**: Grid coordinates
   - Shape: `(2, ny, nx)`
   - `[0, :, :]`: Longitude values
   - `[1, :, :]`: Latitude values

3. **`bathymetry.npy`**: Bathymetry data
   - Shape: `(ny, nx)`
   - Ocean floor depth/elevation values

### Reading Output Data

```python
import numpy as np
from clawpack.pyclaw import Solution

# Load grid and bathymetry
coords = np.load("_output/coord_meshgrid.npy")
lon = coords[0, :, :]
lat = coords[1, :, :]
bathymetry = np.load("_output/bathymetry.npy")

# Load a specific solution frame
solution = Solution()
solution.read(frame=10, path="_output", file_format='pkl')

# Extract variables
h = solution.q[0, :, :]   # Water depth
hu = solution.q[1, :, :]  # x-momentum
hv = solution.q[2, :, :]  # y-momentum

# Calculate velocities (where h > 0)
u = np.where(h > 1e-6, hu / h, 0)
v = np.where(h > 1e-6, hv / h, 0)

# Calculate free surface elevation
eta = h + bathymetry
```

---

## Tips and Best Practices

### 1. Grid Resolution

- **Start coarse**: Begin with `nx=40, ny=40` for testing
- **Refine gradually**: Double resolution to see convergence
- **Memory consideration**: Memory usage scales as `O(nx * ny)`

### 2. Time Step Selection

- **CFL condition**: `dt` should satisfy the Courant-Friedrichs-Lewy condition
- **Rule of thumb**: `dt ≤ min(dx, dy) / sqrt(g * max_depth)`
- **Start conservative**: Use smaller `dt` initially

### 3. Boundary Conditions

- **Ocean boundaries**: Use `extrap` for open ocean boundaries
- **Coast/walls**: Use `wall` for solid boundaries
- **Avoid artifacts**: Match physical boundaries to BC types

### 4. Initial Conditions

- **Dry cells**: Set `eta <= bathymetry` for initially dry regions
- **Smooth transitions**: Avoid discontinuities in initial conditions
- **Balance**: Ensure momentum is consistent with water depth

### 5. Wind Forcing

- **Units**: Wind velocities in m/s
- **Direction convention**: 
  - U > 0: Eastward wind
  - V > 0: Northward wind
- **Magnitude**: Typical values 5-30 m/s for storms

### 6. Coordinate Systems

- **Access grids**: Use `solver.X_coord`, `solver.Y_coord` for lon/lat
- **Metric space**: Use `solver.mapper.coord_to_metric()` for distances
- **Grid alignment**: Arrays are in `(ny, nx)` order (row-major)

### 7. MPI Parallelization

```bash
# Run with MPI
mpirun -n 4 python your_script.py
```

- Automatically parallelized if MPI is available
- Check rank with `solver.rank` for rank-specific operations
- Only rank 0 should do I/O operations

### 8. Debugging

- **Check array shapes**: Verify `(ny, nx)` or `(3, ny, nx)` dimensions
- **Validate configuration**: Errors are raised before solving
- **Inspect output**: Check `pyclaw.log` for solver details
- **Visualize bathymetry**: Plot bathymetry before running simulation

### 9. Performance Optimization

- **Reduce output**: Set `multiple_output_times=False` for final state only
- **Optimize nx/ny**: Balance accuracy vs. computation time
- **Use appropriate dt**: Larger stable dt = faster simulation

---

## Error Messages

### Common Errors and Solutions

**"Domain not set. Call set_domain() first."**
- Solution: Call `solver.set_domain()` before `setup_solver()` or `solve()`

**"Initial condition not set. Call set_initial_condition() first."**
- Solution: Call `solver.set_initial_condition()` with proper array

**"Bathymetry not set. Call set_bathymetry() first."**
- Solution: Call `solver.set_bathymetry()` with bathymetry array

**"Bathymetry array must match grid dimensions"**
- Solution: Ensure bathymetry shape is `(ny, nx)`, not `(nx, ny)`

**"Initial condition array must match grid dimensions"**
- Solution: Ensure initial condition shape is `(3, ny, nx)`

---

## References

- **Clawpack**: http://www.clawpack.org/
- **PyClaw Documentation**: http://www.clawpack.org/pyclaw/
- **Shallow Water Equations**: Classical 2D SWE with bathymetry
- **Riemann Solver**: Uses `sw_aug_2D` (augmented shallow water)

---

## License

This documentation is provided as-is for the sweSolver class.

---

## Contact & Support

For issues, questions, or contributions, please refer to the project repository or contact the development team.
