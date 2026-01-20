#!/usr/bin/env python
# encoding: utf-8
r"""
2D shallow water: radial dam break (PyClaw + Matplotlib)
"""

import matplotlib.pyplot as plt
import numpy as np
from clawpack import riemann
from clawpack.riemann.shallow_roe_with_efix_2D_constants import (
    depth,
    num_eqn,
    x_momentum,
    y_momentum,
)
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # this process ID
size = comm.Get_size()
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from forcing import wind_forcing_step

print("depth =", depth, num_eqn)

x_domain = [-100, 100]
nx = 13
ny = nx
y_domain = [0, 60]
gravity = 10.0
T = 5.0
dt = T / 100
# from petsc4py import PETSc

# opts = PETSc.Options()
# opts["ksp_type"] = "gmres"
# opts["pc_type"] = "lu"
# opts["ksp_rtol"] = 1e-8
# opts["ksp_max_it"] = 1
# print("ksp_type =", opts.getString("ksp_type"))
# print("pc_type  =", opts.getString("pc_type"))
# print("ksp_rtol =", opts.getReal("ksp_rtol"))


def initial_condition(x, y):
    return 2.0 + 1.0 * np.exp(-0.01 * ((x - 50) ** 2))


def bathymetry(x, y):
    return -0.1 * np.abs(x) + 5


def initialize_state(state):
    X, Y = state.p_centers
    surface_elevation = np.maximum(0.0, initial_condition(X, Y) - bathymetry(X, Y))
    state.q[depth, :, :] = surface_elevation
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_zlim(0, 4.5)
    ax.plot_surface(X, Y, surface_elevation)
    ax.plot_surface(X, Y, bathymetry(X, Y), color="brown", alpha=0.5)
    plt.show()
    state.q[x_momentum, :, :] = 0.0
    state.q[y_momentum, :, :] = 0.0

    state.aux[:, :, :] = bathymetry(X, Y)


def problem_setup():
    import clawpack.petclaw as pyclaw
    # import clawpack.pyclaw as pyclaw
    # from clawpack.petclaw import parallel

    # comm = parallel.comm
    # rank = comm.rank
    # size = comm.size

    # if rank == 0:
    #     print(f"Running with {size} MPI ranks")

    rs = riemann.sw_aug_2D

    solver = pyclaw.ClawSolver2D(rs)
    solver.dimensional_split = True
    solver.limiters = pyclaw.limiters.tvd.minmod
    solver.cfl_max = 0.45
    solver.cfl_desired = 0.4
    # solver.kernel_language = "Python"

    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.bc_lower[1] = pyclaw.BC.extrap
    solver.bc_upper[1] = pyclaw.BC.wall

    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[1] = pyclaw.BC.extrap
    solver.aux_bc_upper[1] = pyclaw.BC.periodic

    solver.fwave = True

    # Domain:

    x = pyclaw.Dimension(*x_domain, nx, name="x")
    y = pyclaw.Dimension(*y_domain, ny, name="y")
    domain = pyclaw.Domain([x, y])

    state = pyclaw.State(domain, num_eqn, num_aux=1)

    # Gravitational constant
    state.problem_data["grav"] = gravity
    # state.aux_global["bathymetry_index"] = 0
    # Add wind forcing
    solver.step_source = wind_forcing_step
    solver.source_split = "godunov"
    # solver.sea_level = 0.0
    # solver.dry_tolerance = 1.0e-3

    initialize_state(state)

    claw = pyclaw.Controller()
    claw.verbosity = 0
    claw.tfinal = T
    claw.solution = pyclaw.Solution(state, domain, outdir="_output")
    claw.solver = solver
    # Disable writing any Clawpack output files
    claw.num_output_times = int(T / dt)
    claw.keep_copy = True

    return claw


if __name__ == "__main__":
    # Run the simulation once and plot with Matplotlib instead of saving files
    import matplotlib.pyplot as plt

    claw = problem_setup()
    claw.run()
    sol = claw.frames[0]
    state = sol.state
    q_local = state.q
    num_eqn, nx_local, ny_local = q_local.shape
    x, y = sol.state.grid.p_centers
    print(
        f"Rank {comm.rank}: local q shape = {q_local.shape}, nx_local = {x[0, 0]}, ny_local = {y[0, 0]}"
    )

    if comm.rank == 0:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Get final solution and plot water depth
        sol = claw.frames[0]
        grid = sol.state.grid
        print(sol.state.problem_data)
        print(type(grid))
        X, Y = grid.p_centers
        solutions = np.empty((len(claw.frames), *X.shape))

        print(f"Total number of frames: {len(claw.frames)}")

        for i, sol in enumerate(claw.frames):
            grid = sol.state.grid
            h = sol.q[depth, :, :]
            solutions[i] = h

        # Set z limits before plotting loop
        bath = bathymetry(X, Y)
        for i, h in enumerate(solutions):
            surface = np.maximum(h + bath, bath)
            ax.clear()
            ax.set_zlim(0, 4.5)
            ax.plot_surface(X, Y, surface)
            ax.plot_surface(X, Y, bath, color="brown", alpha=0.5)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            ax.set_title(f"Water surface and bathymetry at {i * dt}")
            plt.pause(1e-3)

        plt.show()
