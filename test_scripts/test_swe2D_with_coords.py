#!/usr/bin/env python
# encoding: utf-8
r"""
2D shallow water: radial dam break (PyClaw + Matplotlib)
"""

import functools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import utils.utils as utils
from clawpack import riemann
from clawpack.riemann.shallow_roe_with_efix_2D_constants import (
    depth,
    num_eqn,
    x_momentum,
    y_momentum,
)
from mpi4py import MPI

from swe_simulator.forcing import wind_forcing_step
from swe_simulator.mapper import LocalLonLatMetricMapper

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # this process ID
size = comm.Get_size()

# Constants and domain setup
lon_min, lon_max = -80.2015, -80.0641
lat_min, lat_max = 25.6528, 25.9287
offset = 0.015
x_domain = [lon_min + offset, lon_max - offset]
nx = 40
ny = nx
y_domain = [lat_min + offset, lat_max - offset]
gravity = 9.81
T = 1000
dt = 1

mapper = LocalLonLatMetricMapper((lon_min + lon_max) / 2, (lat_min + lat_max) / 2)
mins = mapper.coord_to_metric(x_domain[0], y_domain[0])
maxs = mapper.coord_to_metric(x_domain[1], y_domain[1])
x_domain = [float(mins[0]), float(maxs[0])]
y_domain = [float(mins[1]), float(maxs[1])]

bathymetry_interpolator = functools.partial(
    utils.interpolate_gebco_on_grid,
    nc_path="data/gebco_2025_n25.9288_s25.6527_w-80.2016_e-80.0642.nc",
)


def bathymetry(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute bathymetry values for given p-centers  (in meters)."""
    x, y = mapper.metric_to_coord(x, y)
    bathymetry_values = bathymetry_interpolator(X=x, Y=y)
    bathymetry_values[np.isnan(bathymetry_values)] = 0.0
    return bathymetry_values


def initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Define the initial condition for the water surface."""
    return 0.2 + 3 * np.exp(-0.00001 * ((x - 3500) ** 2 + (y + 0) ** 2))


def initialize_state(state) -> None:
    """Initialize the simulation state."""
    X, Y = state.p_centers
    bath = bathymetry(X, Y)
    X_coord, Y_coord = mapper.metric_to_coord(X, Y)
    if os.path.exists("_output") is False and comm.rank == 0:
        os.makedirs("_output")
    np.save("_output/coord_meshgrid.npy", np.stack((X_coord, Y_coord), axis=0))
    np.save("_output/bathymetry.npy", bath)
    surface_elevation = np.maximum(0.0, initial_condition(X, Y) - bath)
    state.q[depth, :, :] = surface_elevation

    # Plot initial state
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(X, Y, surface_elevation)
    ax.set_title("Bathymetry (m), close this to continue")
    fig.colorbar(im, ax=ax)
    plt.show()

    state.q[x_momentum, :, :] = 0.0
    state.q[y_momentum, :, :] = 0.0
    state.aux[:, :, :] = bath


def problem_setup():
    """Set up the PyClaw problem."""
    import clawpack.petclaw as pyclaw

    rs = riemann.sw_aug_2D
    solver = pyclaw.ClawSolver2D(rs)

    # Boundary conditions
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.bc_lower[1] = pyclaw.BC.extrap
    solver.bc_upper[1] = pyclaw.BC.wall

    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[1] = pyclaw.BC.extrap
    solver.aux_bc_upper[1] = pyclaw.BC.extrap

    solver.fwave = True
    solver.step_source = wind_forcing_step
    solver.source_split = 2

    # Domain and state
    x = pyclaw.Dimension(*x_domain, nx, name="x")
    y = pyclaw.Dimension(*y_domain, ny, name="y")
    domain = pyclaw.Domain([x, y])
    state = pyclaw.State(domain, num_eqn, num_aux=1)
    state.problem_data["grav"] = gravity

    initialize_state(state)

    claw = pyclaw.Controller()
    claw.tfinal = T
    claw.solution = pyclaw.Solution(state, domain, outdir="_output")
    claw.solver = solver
    claw.num_output_times = int(T / dt)
    claw.keep_copy = True

    return claw


if __name__ == "__main__":
    claw = problem_setup()
    claw.run()
    print("Last dt used:", claw.solver.dt)

    if comm.rank == 0:
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111)

        # Get final solution and plot water depth
        sol = claw.frames[0]
        grid = sol.state.grid
        X, Y = grid.p_centers

        solutions = np.empty((len(claw.frames), *X.shape))
        for i, sol in enumerate(claw.frames):
            h = sol.q[depth, :, :]
            solutions[i] = h

        result = utils.read_solutions(
            outdir="_output", frames_list=list(range(len(claw.frames)))
        )
        solutions = result["solutions"]
        (X_coord, Y_coord) = result["meshgrid"]
        solutions = solutions[:, 0, ...]
        bath = result["bathymetry"]

        cbar = fig.colorbar(
            ax.pcolormesh(X_coord, Y_coord, bath, shading="auto"),
            ax=ax,
            label="Bathymetry (m)",
        )
        for i, h in enumerate(solutions):
            surface = np.maximum(h + bath, bath)
            above = surface - bath
            try:
                cbar.remove()
            except Exception as e:
                print(f"Error clearing colorbar: {e}")
            ax.clear()

            free_surface = h + bath
            free_surface[h < 1e-3] = np.nan
            im = ax.pcolormesh(
                X_coord, Y_coord, free_surface, shading="auto", cmap="plasma"
            )
            cbar = fig.colorbar(im, ax=ax, label="Elevation (m)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            ax.set_title(f"Water surface and bathymetry at {i * dt}")
            plt.pause(1e-3)

        plt.show()
