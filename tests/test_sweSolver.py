#!/usr/bin/env python
# encoding: utf-8
"""
Test for sweSolver class using the radial dam break example from swe2D_with_coords.py
"""

import functools
import os
import sys

import clawpack.petclaw as pyclaw
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import sweSolver
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils
from mapper import LocalLonLatMetricMapper
from sweSolver import _generate_cell_centers, sweSolver

# Functions to generate bathymetry and initial conditions

bathymetry_interpolator = functools.partial(
    utils.interpolate_gebco_on_grid,
    nc_path="data/gebco_2025_n25.9288_s25.6527_w-80.2016_e-80.0642.nc",
)


def get_bathymetry(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute bathymetry values for given p-centers  (in meters)."""
    # x, y = mapper.metric_to_coord(x, y)
    bathymetry_values = bathymetry_interpolator(X=x, Y=y)
    bathymetry_values[np.isnan(bathymetry_values)] = 0.0
    return bathymetry_values


def get_initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Define the initial condition for the water surface."""
    # x, y = mapper.coord_to_metric(x, y)
    tide_height = 0.2 + 3 * np.exp(-0.00001 * ((x - 3500) ** 2 + (y + 0) ** 2))
    m_x = np.zeros_like(tide_height)
    m_y = np.zeros_like(tide_height)
    return np.stack([tide_height, m_x, m_y], axis=0)


def get_wind_forcing() -> tuple[float, float]:
    """Define wind forcing parameters."""
    speed_florida = 57  # mph
    U_a = (-1 / np.sqrt(2)) * 0.44 * speed_florida  # m/s
    V_a = (1 / np.sqrt(2)) * 0.44 * speed_florida  # m/s
    return U_a, V_a


def test_radial_dam_break() -> None:
    """Test sweSolver with radial dam break scenario from swe2D_with_coords.py"""

    # Constants and domain setup
    lon_min, lon_max = -80.2015, -80.0641
    lat_min, lat_max = 25.6528, 25.9287
    offset = 0.015
    lon_range = (lon_min + offset, lon_max - offset)
    lat_range = (lat_min + offset, lat_max - offset)

    # Number of grid cells
    nx = 40
    ny = 40

    # Gravity and time parameters
    gravity = 9.81
    T = 1000
    dt = 1

    mapper = LocalLonLatMetricMapper((lon_min + lon_max) / 2, (lat_min + lat_max) / 2)

    # Generate grid cell centers (Domain will be set in solver)
    X_coord, Y_coord = _generate_cell_centers(
        lon_range[0], lon_range[1], lat_range[0], lat_range[1], nx, ny
    )
    """
        X_coord has shape (nx, ny)
        Y_coord has shape (nx, ny)
        bathymetry_values has shape (nx, ny)
        initial_surface has shape (3, nx, ny)  # 3 for [tide_height, momentum_x, momentum_y]
        boundary conditions:
            lower: has shape (2,)  # lower: [wall (lon), extrap (lat)]
            upper: has shape (2,)  # upper: [extrap (lon), wall (lat)]
    """
    # Get bathymetry values
    bathymetry_values = get_bathymetry(X_coord, Y_coord)
    # Get initial condition values, tide height, momentum_x, momentum_y (we keep momentum zero here)
    initial_surface = get_initial_condition(*mapper.coord_to_metric(X_coord, Y_coord))
    wind_forcing_vector = get_wind_forcing()

    # Initialize solver
    solver = sweSolver(True)
    # Set domain
    solver.set_domain(lon_range=lon_range, lat_range=lat_range, nx=nx, ny=ny)
    solver.gravity = gravity

    # Set time parameters
    solver.set_time_parameters(t_final=T, dt=dt)

    # Set bathymetry
    solver.set_bathymetry(bathymetry_array=bathymetry_values)

    # Set initial condition
    solver.set_initial_condition(initial_condition=initial_surface)

    # Set boundary conditions you can leave all wall or all extrap
    solver.set_boundary_conditions(
        lower=[pyclaw.BC.wall, pyclaw.BC.extrap],
        upper=[pyclaw.BC.extrap, pyclaw.BC.wall],
    )
    solver.set_forcing(wind_vector=wind_forcing_vector)

    # Setup and run solver
    solver.setup_solver()
    solver.solve()

    print(f"Last dt used: {solver.dt}")

    # Visualize results (only on rank 0 for MPI)
    if solver.rank == 0:
        # Read solutions
        result = utils.read_solutions(
            outdir="_output", frames_list=list(range(len(solver.claw.frames)))
        )
        solutions = result["solutions"][:, 0, ...]  # Extract depth component
        (X_coord, Y_coord) = result["meshgrid"]
        bath = result["bathymetry"]

        # Create animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Plot bathymetry
        cbar = fig.colorbar(
            ax.pcolormesh(X_coord, Y_coord, bath, shading="auto"),
            ax=ax,
            label="Bathymetry (m)",
        )

        # Animate water surface
        for i, h in enumerate(solutions):
            try:
                cbar.remove()
            except Exception as e:
                print(f"Error clearing colorbar: {e}")
            ax.clear()

            # Calculate free surface elevation
            free_surface = h + bath
            free_surface[h < 1e-3] = np.nan

            im = ax.pcolormesh(
                X_coord, Y_coord, free_surface, shading="auto", cmap="plasma"
            )
            cbar = fig.colorbar(im, ax=ax, label="Elevation (m)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal")
            ax.set_title(f"Water surface and bathymetry at t={i * dt}s")
            plt.pause(0.01)

        plt.show()

        print("Test completed successfully!")


if __name__ == "__main__":
    test_radial_dam_break()
