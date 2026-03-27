#!/usr/bin/env python
# encoding: utf-8
"""Simple example: Gaussian hump in flat bathymetry."""

import logging

import clawpack.petclaw as pyclaw

import tidalflow

logger = tidalflow.logging_config.setup_logging(
    logging.INFO,
    "gaussian_hump_example.log",
)


def run_gaussian_hump_example() -> None:
    """
    Simple example: Gaussian hump in flat bathymetry.

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
        dt=0.25,  # seconds
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

    logger.info("Creating providers...")

    # Flat bathymetry at 1m depth
    bathymetry_provider = tidalflow.providers.FlatBathymetry(depth=-1.0)

    # Gaussian hump
    initial_condition_provider = tidalflow.providers.GaussianHumpInitialConditionNoGeo(
        bias=0.2,
        height=3.0,  # 1 meter hump
        width=100.0,  # width parameter (larger = wider hump)
        center=(0.25, 0.5),  # Center of the hump in normalized coordinates (0 to 1)
    )

    # Solver setup

    logger.info("Initializing SWESolver...")
    solver = tidalflow.solver.SWESolver(
        config=config,
        bathymetry_provider=bathymetry_provider,
        ic_provider=initial_condition_provider,
    )

    logger.info(f"Configuration:\n{solver.config}")

    logger.info("Initializing data from providers...")
    solver.initialize_data_from_providers()

    logger.info(
        f"Bathymetry: {solver.bathymetry_array.min():.2f}m"
        f" to {solver.bathymetry_array.max():.2f}m"
    )
    logger.info(
        f"Initial water height: {solver.initial_condition_array[0].min():.2f}m "
        f"to {solver.initial_condition_array[0].max():.2f}m"
    )

    # Run simulation

    logger.info("Setting up solver...")
    solver.setup_solver()

    logger.info("Running simulation...")
    result = solver.solve()
    assert result.solution is not None

    logger.info(f"Simulation complete!")
    logger.info(f"Solution shape: {result.solution.shape}")
    logger.info(f"Number of output frames: {len(result.solution)}")

    # Visualize results

    if solver.rank == 0 and solver.config.output_dir is not None:
        mpl_rc_params = {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Google Sans", "DejaVu Sans"],
        }
        frames = None  # All frames
        wave_threshold = 1e-3
        interval = 50
        save = True
        dark_mode = True
        file_name_surface = "gaussian_hump.gif"
        file_name_velocity = "gaussian_hump_velocity.gif"
        writer = "pillow"
        figsize_surface = (5, 4)
        figsize_velocity = (7, 5)
        fps = 20
        logger.info("Animating results...")
        tidalflow.utils.visualization.animate_surface(
            output_path=solver.config.output_dir,
            frames=frames,  # All frames
            wave_treshold=wave_threshold,
            interval=interval,
            save=save,
            dark_mode=dark_mode,
            file_name=file_name_surface,
            writer=writer,
            figsize=figsize_surface,
            mpl_rc_params=mpl_rc_params,
            fps=fps,
        )
        tidalflow.utils.visualization.animate_solution(
            output_path=solver.config.output_dir,
            frames=frames,  # All frames
            wave_treshold=wave_threshold,
            interval=interval,
            save=save,
            dark_mode=dark_mode,
            file_name=file_name_velocity,
            writer=writer,
            figsize=figsize_velocity,
            mpl_rc_params=mpl_rc_params,
            arrow_step=5,
            fps=fps,
        )
        logger.info("Visualization complete!")

    logger.info("Example completed successfully!")


if __name__ == "__main__":
    run_gaussian_hump_example()
