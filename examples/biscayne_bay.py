#!/usr/bin/env python
# encoding: utf-8
"""Test for SWESolver class using the radial dam break example."""

import logging

import clawpack.petclaw as pyclaw
import numpy as np

import tidalflow

logger = tidalflow.logging_config.setup_logging(
    logging.DEBUG,
    "biscayne_bay_example.log",
)


def test_radial_dam_break() -> None:
    """Test SWESolver with radial dam break scenario."""

    # Configuration

    # Domain bounds
    lon_min, lon_max = -80.2015, -80.0641
    lat_min, lat_max = 25.6528, 25.9287
    offset = 0.015
    lon_range = (lon_min + offset, lon_max - offset)
    lat_range = (lat_min + offset, lat_max - offset)

    # Create configuration
    config = tidalflow.config.SimulationConfig(
        # Domain
        lon_range=lon_range,
        lat_range=lat_range,
        nx=40,
        ny=40,
        # Time
        t_final=1000.0,  # seconds
        dt=10.0,  # seconds
        # Physics
        gravity=9.81,
        # Boundary conditions
        bc_lower=(pyclaw.BC.extrap, pyclaw.BC.extrap),
        bc_upper=(pyclaw.BC.extrap, pyclaw.BC.extrap),
        # Output
        output_dir="output_biscayne_bay",
        multiple_output_times=True,  # Will use t_final/dt
    )

    # Wind parameters (Hurricane-like conditions)
    speed_florida = 57  # mph
    u_wind = (-1 / np.sqrt(2)) * 0.44 * speed_florida  # m/s
    v_wind = (1 / np.sqrt(2)) * 0.44 * speed_florida  # m/s

    # Providers

    print("Creating data providers...")

    # Bathymetry from GEBCO NetCDF file
    bathymetry_provider = tidalflow.providers.BathymetryFromNC(
        nc_path="data/gebco_2025_n25.9288_s25.6527_w-80.2016_e-80.0642.nc"
    )

    # Initial condition: Gaussian hump centered at domain center (in geographic coords)
    # Domain center in lon/lat
    alpha_lon = 0.25
    alpha_lat = 0.6
    center_lon = alpha_lon * lon_range[0] + (1 - alpha_lon) * lon_range[1]
    center_lat = alpha_lat * lat_range[0] + (1 - alpha_lat) * lat_range[1]
    print(f"Domain center (lon, lat): ({center_lon:.4f}, {center_lat:.4f})")
    initial_condition_provider = tidalflow.providers.GaussianHumpInitialCondition(
        height=3,  # meters
        width=10000,  # controls spread in coordinate space (roughly 1 degree ~ 111111 m,
        bias=0.25,  # base water level (tide)
        center=(center_lon, center_lat),
    )

    # Solver setup

    print("Initializing SWESolver...")
    solver = tidalflow.solver.SWESolver(
        config=config,
        bathymetry_provider=bathymetry_provider,
        ic_provider=initial_condition_provider,
    )

    print(f"Config:\n {config}")

    # Initialize data from providers

    print("Initializing data from providers...")
    solver.initialize_data_from_providers()
    print(
        f"Bathymetry: min={solver.bathymetry_array.min():.2f}m, "
        f"max={solver.bathymetry_array.max():.2f}m"
    )
    print(
        f"Initial water depth: min={solver.initial_condition_array[0].min():.2f}m, "
        f"max={solver.initial_condition_array[0].max():.2f}m"
    )

    print(f"Boundary conditions: lower={config.bc_lower}, upper={config.bc_upper}")

    # Set wind forcing (direct values, not provider)

    print(f"Setting wind forcing: u={u_wind:.2f} m/s, v={v_wind:.2f} m/s")
    solver.set_constant_wind_forcing(u_wind=u_wind, v_wind=v_wind)

    # Run simulation

    print("Setting up solver...")
    solver.setup_solver()

    print("Running simulation...")
    result = solver.solve()
    assert result.solution is not None

    print(
        f"\nSimulation complete! solution tensor (T+1, 3, nx, ny): {result.solution.shape}"
    )

    # Visualize results (only on rank 0 for MPI)

    if solver.rank == 0 and solver.config.output_dir is not None:
        tidalflow.utils.visualization.animate_solution(
            output_path=solver.config.output_dir,
            frames=None,  # It means all frames
            wave_treshold=1e-2,
            save=False,
            dark_mode=True,
            writer="pillow",
            file_name="biscayne_bay.gif",
            fps=25,
        )
        print("\nVisualization complete!")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_radial_dam_break()
