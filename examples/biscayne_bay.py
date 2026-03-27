#!/usr/bin/env python
# encoding: utf-8
"""Example: Storm surge simulation for Biscayne Bay."""

import logging

import clawpack.petclaw as pyclaw
import numpy as np

import tidalflow

logger = tidalflow.logging_config.setup_logging(
    logging.INFO,
    "biscayne_bay_example.log",
)


def run_radial_dam_break_example() -> None:
    """Example: Storm surge simulation using radial dam break scenario in Biscayne Bay."""

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
        nx=50,
        ny=50,
        # Time
        t_final=100.0,  # seconds
        dt=5.0,  # seconds
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

    logger.info("Creating data providers...")

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
    logger.info(f"Domain center (lon, lat): ({center_lon:.4f}, {center_lat:.4f})")
    initial_condition_provider = tidalflow.providers.GaussianHumpInitialCondition(
        height=3,  # meters
        width=10000,  # controls spread in coordinate space (roughly 1 degree ~ 111111 m,
        bias=0.25,  # base water level (tide)
        center=(center_lon, center_lat),
    )

    # Solver setup

    logger.info("Initializing SWESolver...")
    solver = tidalflow.solver.SWESolver(
        config=config,
        bathymetry_provider=bathymetry_provider,
        ic_provider=initial_condition_provider,
    )

    logger.info(f"Configuration:\n{solver.config}")

    # Initialize data from providers

    logger.info("Initializing data from providers...")
    solver.initialize_data_from_providers()
    logger.info(
        f"Bathymetry: {solver.bathymetry_array.min():.2f}m "
        f"to {solver.bathymetry_array.max():.2f}m"
    )
    logger.info(
        f"Initial water depth: {solver.initial_condition_array[0].min():.2f}m "
        f"to {solver.initial_condition_array[0].max():.2f}m"
    )

    # Set wind forcing (direct values, not provider)

    logger.info(f"Setting wind forcing: u={u_wind:.2f} m/s, v={v_wind:.2f} m/s")
    solver.set_constant_wind_forcing(u_wind=u_wind, v_wind=v_wind)

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
        logger.info("Animating results...")
        mpl_rc_params = {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Google Sans", "DejaVu Sans"],
        }
        frames = None  # All frames
        wave_threshold = 1e-2
        interval = 50
        save = False
        dark_mode = True
        file_name_surface = "biscayne_bay.gif"
        file_name_velocity = "biscayne_bay_velocity.gif"
        writer = "pillow"
        figsize_surface = (6, 5)
        figsize_velocity = (6, 7)
        fps = 20

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
            arrow_step=2,
            fps=fps,
        )
        logger.info("Visualization complete!")

    logger.info("Example completed successfully!")


if __name__ == "__main__":
    run_radial_dam_break_example()
