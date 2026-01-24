#!/usr/bin/env python
# encoding: utf-8
"""Test for SWESolver class using the radial dam break example."""

import functools
import os
import sys

import clawpack.petclaw as pyclaw
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import logging

import swe_simulator
import swe_simulator.utils as sim_utils

logger = logging.getLogger(__name__)
# Helper functions for bathymetry and initial conditions
# ============================================================================

bathymetry_interpolator = functools.partial(
    sim_utils.interpolate_gebco_on_grid,
    nc_path="data/gebco_2025_n25.9288_s25.6527_w-80.2016_e-80.0642.nc",
)


def get_bathymetry(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Compute bathymetry values for given coordinates.

    Parameters
    ----------
    lon : np.ndarray
        Longitude coordinates in degrees
    lat : np.ndarray
        Latitude coordinates in degrees

    Returns
    -------
    np.ndarray
        Bathymetry values in meters (negative = depth)
    """
    bathymetry_values = bathymetry_interpolator(X=lon, Y=lat)
    bathymetry_values[np.isnan(bathymetry_values)] = 0.0
    return bathymetry_values


def get_initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Define initial condition for water surface.

    Parameters
    ----------
    x : np.ndarray
        X coordinates in meters
    y : np.ndarray
        Y coordinates in meters

    Returns
    -------
    np.ndarray
        Initial condition array of shape (3, ny, nx) with [h, hu, hv]
    """
    # Radial dam break: Gaussian hump
    tide_height = 0.2 + 3 * np.exp(-0.00001 * ((x - 3500) ** 2 + (y + 0) ** 2))
    m_x = np.zeros_like(tide_height)
    m_y = np.zeros_like(tide_height)
    return np.stack([tide_height, m_x, m_y], axis=0)


def test_radial_dam_break() -> None:
    """Test SWESolver with radial dam break scenario."""

    # ========================================================================
    # Configuration
    # ========================================================================

    # Domain bounds
    lon_min, lon_max = -80.2015, -80.0641
    lat_min, lat_max = 25.6528, 25.9287
    offset = 0.015
    lon_range = (lon_min + offset, lon_max - offset)
    lat_range = (lat_min + offset, lat_max - offset)

    # Wind parameters (Hurricane-like conditions)
    speed_florida = 57  # mph
    u_wind = (-1 / np.sqrt(2)) * 0.44 * speed_florida  # m/s (convert to m/s)
    v_wind = (1 / np.sqrt(2)) * 0.44 * speed_florida  # m/s

    # Create configuration
    config = swe_simulator.SimulationConfig(
        # Domain
        lon_range=lon_range,
        lat_range=lat_range,
        nx=40,
        ny=40,
        # Time
        t_final=10.0,  # seconds
        dt=1.0,  # seconds
        # Physics
        gravity=9.81,
        # Boundary conditions
        bc_lower=(pyclaw.BC.extrap, pyclaw.BC.extrap),
        bc_upper=(pyclaw.BC.extrap, pyclaw.BC.extrap),
        # Output
        output_dir="_outputs",
        multiple_output_times=True,  # Will use t_final/dt
    )

    # Wind parameters (Hurricane-like conditions)
    speed_florida = 57  # mph
    u_wind = (1 / np.sqrt(2)) * 0.44 * speed_florida  # m/s (convert to m/s)
    v_wind = (-1 / np.sqrt(2)) * 0.44 * speed_florida  # m/s

    # ========================================================================
    # Initialize Solver
    # ========================================================================

    print("Initializing SWESolver...")
    solver = swe_simulator.SWESolver(config=config)
    # solver._validate_configuration()

    # # You can set domain individually if desired as
    # solver.set_domain(lon_range=config.lon_range, lat_range=config.lat_range, nx=config.nx, ny=config.ny)
    # solver.gravity = config.gravity

    # # Set time parameters individually if desired as well
    # solver.set_time_parameters(t_final=config.t_final, dt=config.dt)

    # # Set boundary conditions individually if desired as well
    # solver.set_boundary_conditions(
    #     lower=(pyclaw.BC.wall, pyclaw.BC.extrap),  # [x_lower, y_lower]
    #     upper=(pyclaw.BC.extrap, pyclaw.BC.wall),  # [x_upper, y_upper]
    # )

    print(f"Domain: lon={config.lon_range}, lat={config.lat_range}")
    print(f"Grid: {config.nx}x{config.ny} cells")
    print(f"Time: t_final={config.t_final}s, dt={config.dt}s")

    # ========================================================================
    # Set Bathymetry
    # ========================================================================

    print("Loading bathymetry...")
    # Generating bathymetry on solver grid
    lon_coords, lat_coords = solver.X_coord, solver.Y_coord

    bathymetry_values = get_bathymetry(lon_coords, lat_coords)
    # Set bathymetry in solver
    solver.set_bathymetry(bathymetry_array=bathymetry_values)

    print(
        f"Bathymetry: min={bathymetry_values.min():.2f}m, max={bathymetry_values.max():.2f}m"
    )

    # ========================================================================
    # Set Initial Condition
    # ========================================================================

    print("Setting initial condition...")
    # Get metric coordinates for initial condition

    # Getting initial condition on solver grid
    initial_surface = get_initial_condition(
        *solver.mapper.coord_to_metric(lon_coords, lat_coords)
    )
    # Set initial condition in solver
    solver.set_initial_condition(initial_condition=initial_surface)

    print(
        f"Initial water depth: min={initial_surface[0].min():.2f}m, "
        f"max={initial_surface[0].max():.2f}m"
    )

    print(f"Boundary conditions: lower={config.bc_lower}, upper={config.bc_upper}")

    # ========================================================================
    # Set Wind Forcing
    # ========================================================================

    print(f"Setting wind forcing: u={u_wind:.2f} m/s, v={v_wind:.2f} m/s")
    solver.set_wind_forcing(u_wind=u_wind, v_wind=v_wind)

    # ========================================================================
    # Setup and Run Solver
    # ========================================================================

    print("\nSetting up solver...")
    solver.setup_solver()

    print("Running simulation...")
    solutions = solver.solve()  # If the solutions are needed for post-processing

    print(f"\nSimulation complete! solution tensor (T, 3, nx, ny): {solutions.shape}")
    print(f"Last dt used: {solver.config.dt}")

    # ========================================================================
    # Visualize Results (only on rank 0 for MPI)
    # ========================================================================

    if solver.rank == 0 and solver.config.output_dir is not None:
        swe_simulator.utils.animate_solution(
            output_path=solver.config.output_dir,
            frames=None,  # It means all frames
            wave_treshold=1e-2,
            interval=100,
            save=False,
        )
        print("\nVisualization complete!")

    print("\nTest completed successfully!")


# def test_simple_example() -> None:
#     """Simple test case without external data dependencies."""

#     print("Running simple test case...")

#     # Simple domain
#     solver = SWESolver(multiple_output_times=True)
#     solver.set_domain(lon_range=(-1.0, 1.0), lat_range=(-1.0, 1.0), nx=50, ny=50)
#     solver.set_time_parameters(t_final=10.0, dt=0.1)

#     # Flat bathymetry at -10m
#     bathymetry = -10.0 * np.ones((solver.ny, solver.nx))
#     solver.set_bathymetry(bathymetry)

#     # Gaussian hump initial condition
#     lon_coords, lat_coords = solver.X_coord, solver.Y_coord
#     x, y = solver.mapper.coord_to_metric(lon_coords, lat_coords)
#     h_init = 2.0 * np.exp(-0.01 * (x**2 + y**2))
#     initial_condition = np.array([h_init, np.zeros_like(h_init), np.zeros_like(h_init)])
#     solver.set_initial_condition(initial_condition)

#     # Set boundary conditions
#     solver.set_boundary_conditions(
#         lower=[pyclaw.BC.wall, pyclaw.BC.wall],
#         upper=[pyclaw.BC.wall, pyclaw.BC.wall],
#     )

#     # Add wind forcing
#     solver.set_wind_forcing(u_wind=5.0, v_wind=3.0)

#     # Run
#     solver.setup_solver()
#     solver.solve()

#     print(f"Simple test complete! Frames generated: {len(solver.claw.frames)}")


if __name__ == "__main__":
    # Run the main test
    test_radial_dam_break()

    # Optionally run simple test
    # test_simple_example()
