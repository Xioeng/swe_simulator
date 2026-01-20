#!/usr/bin/env python
# encoding: utf-8
r"""
2D shallow water: radial dam break (PyClaw + Matplotlib)
"""

import numpy as np
from clawpack import riemann
from clawpack.riemann.shallow_roe_with_efix_2D_constants import (
    depth,
    num_eqn,
    x_momentum,
    y_momentum,
)

print("depth =", depth)


def qinit(state, h_in=2.0, h_out=1.0, dam_radius=0.5):
    x0 = 0.0
    y0 = 0.0
    X, Y = state.p_centers
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)

    state.q[depth, :, :] = h_in * (r <= dam_radius) + h_out * (r > dam_radius)
    state.q[x_momentum, :, :] = 0.0
    state.q[y_momentum, :, :] = 0.0


def setup(
    kernel_language="Fortran",
    use_petsc=True,
    outdir="./_output",
    solver_type="classic",
    riemann_solver="roe",
    disable_output=False,
):
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if riemann_solver.lower() == "roe":
        rs = riemann.shallow_roe_with_efix_2D
    elif riemann_solver.lower() == "hlle":
        rs = riemann.shallow_hlle_2D

    if solver_type == "classic":
        solver = pyclaw.ClawSolver2D(rs)
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.dimensional_split = 1
    elif solver_type == "sharpclaw":
        solver = pyclaw.SharpClawSolver2D(rs)

    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.bc_lower[1] = pyclaw.BC.extrap
    solver.bc_upper[1] = pyclaw.BC.wall

    # Domain:
    xlower = -2.5
    xupper = 2.5
    mx = 150
    ylower = -2.5
    yupper = 2.5
    my = 150
    x = pyclaw.Dimension(xlower, xupper, mx, name="x")
    y = pyclaw.Dimension(ylower, yupper, my, name="y")
    domain = pyclaw.Domain([x, y])

    state = pyclaw.State(domain, num_eqn)

    # Gravitational constant
    state.problem_data["grav"] = 1.0

    qinit(state)

    claw = pyclaw.Controller()
    claw.tfinal = 4.0
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    # Disable writing any Clawpack output files
    claw.output_format = None
    claw.outdir = outdir
    claw.num_output_times = 100
    claw.keep_copy = True

    return claw


if __name__ == "__main__":
    # Run the simulation once and plot with Matplotlib instead of saving files
    import matplotlib.pyplot as plt

    claw = setup(use_petsc=True, disable_output=True, solver_type="sharpclaw")
    claw.run()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Get final solution and plot water depth
    sol = claw.frames[0]
    grid = sol.state.grid
    X, Y = grid.p_centers
    solutions = np.empty((len(claw.frames), *X.shape))

    for i, sol in enumerate(claw.frames):
        grid = sol.state.grid

        h = sol.q[depth, :, :]
        solutions[i] = h

    # Set z limits before plotting loop
    for h in solutions:
        ax.clear()
        ax.set_zlim(0, 2.5)
        ax.plot_surface(X, Y, h)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.pause(1e-3)

    plt.show()
