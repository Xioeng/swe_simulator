#!/usr/bin/env python
# encoding: utf-8
"""Simple example: Gaussian hump in flat bathymetry."""

import logging

import clawpack.petclaw as pyclaw
import numpy as np

import tidalflow

logger = tidalflow.logging_config.setup_logging(
    logging.DEBUG,
    "gaussian_hump_example.log",
)


def test_gaussian_hump() -> None:
    """
    Simple test: Gaussian hump in flat bathymetry.

    Domain: approximately 100m x 100m (0.0009 degrees x 0.0009 degrees)
    Grid: 100 x 100 cells
    Bathymetry: flat at -1m
    Initial condition: Gaussian hump centered at domain center
    """

    # Configuration (domain ~100m x 100m)

    # At the equator, 1 degree ≈ 111 km = 111,000 m
    # So 100m ≈ 0.0009 degrees
    # Use ±0.0005 degrees for simplicity (~111m span total)
    max_lon = 0.0005  # half-width in longitude (degrees)
    max_lat = 0.0005  # half-width in latitude (degrees)

    lon_range = (-max_lon, max_lon)
    lat_range = (-max_lat, max_lat)

    config = tidalflow.config.SimulationConfig(
        # Domain
        lon_range=lon_range,
        lat_range=lat_range,
        nx=100,
        ny=100,
        # Time stepping
        t_final=50.0,  # seconds
        dt=0.5,  # seconds
        # Physics
        gravity=9.81,
        # Boundary conditions (wall on all sides)
        bc_lower=(pyclaw.BC.wall, pyclaw.BC.wall),
        bc_upper=(pyclaw.BC.wall, pyclaw.BC.wall),
        # Output
        output_dir="output_gaussian_hump",
        multiple_output_times=True,
    )

    # Providers

    print("Creating providers...")

    # Flat bathymetry at 1m depth
    bathymetry_provider = tidalflow.providers.FlatBathymetry(depth=-1.0)

    # Gaussian hump
    initial_condition_provider = (
        tidalflow.providers.GaussianHumpInitialConditionNoGeo(
            bias=0.2,
            height=3.0,  # 1 meter hump
            width=100.0,  # width parameter (larger = wider hump)
            center=(0.25, 0.5),  # Center of the hump in normalized coordinates (0 to 1)
        )
    )

    # Solver setup

    print("Initializing SWESolver...")
    solver = tidalflow.solver.SWESolver(
        config=config,
        bathymetry_provider=bathymetry_provider,
        ic_provider=initial_condition_provider,
    )

    print(f"\nConfiguration:\n{solver.config}")

    print("\nInitializing data from providers...")
    solver.initialize_data_from_providers()

    print(
        f"  Bathymetry: {solver.bathymetry_array.min():.2f}m"
        f" to {solver.bathymetry_array.max():.2f}m"
    )
    print(
        f"  Initial water height: {solver.initial_condition_array[0].min():.2f}m "
        f"to {solver.initial_condition_array[0].max():.2f}m"
    )

    # Run simulation

    print("\nSetting up solver...")
    solver.setup_solver()
    print(f"\n {np.max(solver.X)} {np.min(solver.X)}")

    print("Running simulation...")
    result = solver.solve()
    assert result.solution is not None

    print(f"\nSimulation complete!")
    print(f"  Solution shape: {result.solution.shape}")
    print(f"  Number of output frames: {len(result.solution)}")

    # Visualize results

    if solver.rank == 0 and solver.config.output_dir is not None:
        print("\nAnimating results...")
        tidalflow.utils.visualization.animate_surface(
            output_path=solver.config.output_dir,
            frames=None,  # All frames
            wave_treshold=1e-3,
            interval=50,
            save=False,
            dark_mode=True,
            file_name="gaussian_hump.gif",
            writer="pillow",
        )
        print("Visualization complete!")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_gaussian_hump()
