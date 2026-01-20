#!/usr/bin/env python
# encoding: utf-8


import os
from typing import List, Optional, Tuple

import clawpack.petclaw as pyclaw
import numpy as np
from clawpack import riemann
from clawpack.riemann.shallow_roe_with_efix_2D_constants import (
    depth,
    num_eqn,
    x_momentum,
    y_momentum,
)
from mpi4py import MPI

import forcing
from mapper import LocalLonLatMetricMapper


def _generate_cell_centers(
    x_lower: float, x_upper: float, y_lower: float, y_upper: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cell center coordinates for a 2D grid.

    This mimics how PyClaw calculates p_centers from domain dimensions.

    Parameters
    ----------
    x_lower : float
        Lower bound of x domain
    x_upper : float
        Upper bound of x domain
    y_lower : float
        Lower bound of y domain
    y_upper : float
        Upper bound of y domain
    nx : int
        Number of cells in x direction
    ny : int
        Number of cells in y direction

    Returns
    -------
    X : np.ndarray
        2D array of shape (ny, nx) with x-coordinates of cell centers
    Y : np.ndarray
        2D array of shape (ny, nx) with y-coordinates of cell centers
    """
    # Calculate cell width
    dx = (x_upper - x_lower) / nx
    dy = (y_upper - y_lower) / ny

    # Generate 1D arrays of cell centers
    x_centers = x_lower + (np.arange(nx) + 0.5) * dx
    y_centers = y_lower + (np.arange(ny) + 0.5) * dy

    # Create 2D meshgrid
    X, Y = np.meshgrid(x_centers, y_centers)

    return X, Y


class sweSolver:
    """
    Solver class for 2D shallow water equations using PyClaw.
    """

    def __init__(self, multiple_output_times: bool = True):
        self.t_final: float = None
        self.dt: float = None
        self.gravity: float = 9.81
        self.comm: MPI.Comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()
        self.multiple_output_times: bool = multiple_output_times

        # Domain configuration
        self.x_domain: Optional[List[float]] = None
        self.y_domain: Optional[List[float]] = None
        self.nx: int = 40
        self.ny: int = 40
        self.mapper: Optional[LocalLonLatMetricMapper] = None

        # Arrays instead of functions
        self.bathymetry_array: Optional[np.ndarray] = None
        self.initial_condition_array: Optional[np.ndarray] = None

        # Boundary Conditions Defaults
        self.bc_lower = [pyclaw.BC.wall, pyclaw.BC.wall]
        self.bc_upper = [pyclaw.BC.wall, pyclaw.BC.wall]

        # Internal state
        self.claw: Optional[pyclaw.Controller] = None

    def set_time_parameters(self, t_final: float, dt: float):
        """
        Set the time parameters for the simulation.

        Args:
            t_final: final time
            dt: time step size
        """
        self.t_final = t_final
        self.dt = dt

    def set_domain(
        self,
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        nx: int,
        ny: int,
    ):
        """
        Set up the grid domain based on longitude and latitude ranges.

        Args:
            lon_range: (min_lon, max_lon)
            lat_range: (min_lat, max_lat)
            nx: number of cells in x direction
            ny: number of cells in y direction
        """
        self.nx = nx
        self.ny = ny

        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range

        self.mapper = LocalLonLatMetricMapper(
            (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
        )
        mins = self.mapper.coord_to_metric(lon_min, lat_min)
        maxs = self.mapper.coord_to_metric(lon_max, lat_max)

        self.x_domain = [float(mins[0]), float(maxs[0])]
        self.y_domain = [float(mins[1]), float(maxs[1])]
        self.X, self.Y = _generate_cell_centers(
            self.x_domain[0],
            self.x_domain[1],
            self.y_domain[0],
            self.y_domain[1],
            self.nx,
            self.ny,
        )
        self.X_coord, self.Y_coord = self.mapper.metric_to_coord(self.X, self.Y)

    def set_bathymetry(
        self,
        bathymetry_array: np.ndarray,
    ):
        """
        Set the bathymetry for the domain.

        Args:
            bathymetry_array: A numpy array matching (ny, nx) or (nx, ny)
        """
        assert bathymetry_array.shape == (self.ny, self.nx), (
            "Bathymetry array must match grid dimensions"
        )
        self.bathymetry_array = bathymetry_array

    def set_initial_condition(self, initial_condition: np.ndarray):
        """
        Set the initial water surface elevation function.
        Function signature: f(x, y) -> elevation (eta)
        """
        assert initial_condition.shape == (3, self.ny, self.nx), (
            "Initial condition array must match grid dimensions"
        )
        self.initial_condition_array = initial_condition

    def set_boundary_conditions(self, lower: List[int], upper: List[int]):
        """
        Set boundary conditions.

        Args:
            lower: List of 2 BCs for x-lower and y-lower [x_lo, y_lo]
            upper: List of 2 BCs for x-upper and y-upper [x_hi, y_hi]
            0: wall, 1: extrapolate, 2: periodic
        """
        self.bc_lower = lower
        self.bc_upper = upper

    def set_forcing(self, wind_vector: Tuple[float, float]):
        """
        Set wind forcing parameters.

        Args:
            wind_vector: (U_wind, V_wind) in m/s
        """
        self.wind_vector = wind_vector
        U_wind, V_wind = self.wind_vector
        forcing.set_wind(U_wind, V_wind)

    def _validate_configuration(self):
        """
        Validate that all required configuration methods have been called.

        Raises:
            RuntimeError: If any required configuration is missing.
        """
        if self.x_domain is None or self.y_domain is None or self.mapper is None:
            raise RuntimeError("Domain not set. Call set_domain() first.")

        if self.initial_condition_array is None:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition() first."
            )

        if self.bathymetry_array is None:
            raise RuntimeError("Bathymetry not set. Call set_bathymetry() first.")

        if self.t_final is None or self.dt is None:
            raise RuntimeError(
                "Time parameters not set. Call set_time_parameters() first."
            )

    def _initialize_solution_state(self, state):
        X, Y = state.p_centers

        # Ensure output directory exists
        if os.path.exists("_output") is False and self.rank == 0:
            os.makedirs("_output")
        # Save grid metadata
        np.save(
            "_output/coord_meshgrid.npy",
            np.stack((self.X_coord, self.Y_coord), axis=0),
        )
        np.save("_output/bathymetry.npy", self.bathymetry_array)

        # Depth h = max(0, eta - B)
        h = np.maximum(0.0, self.initial_condition_array[0] - self.bathymetry_array)

        state.q[depth, :, :] = h
        state.q[x_momentum, :, :] = self.initial_condition_array[1]
        state.q[y_momentum, :, :] = self.initial_condition_array[2]
        state.aux[:, :, :] = self.bathymetry_array

    def setup_solver(self) -> pyclaw.Controller:
        """
        Constructs the PyClaw solver, domain, and controller based on configuration.
        """
        self._validate_configuration()

        rs = riemann.sw_aug_2D
        solver = pyclaw.ClawSolver2D(rs)

        # Assign BCs
        solver.bc_lower[0] = self.bc_lower[0]
        solver.bc_upper[0] = self.bc_upper[0]
        solver.bc_lower[1] = self.bc_lower[1]
        solver.bc_upper[1] = self.bc_upper[1]

        # Aux BCs (Extrapolate is generally safe for bathymetry in this context)
        solver.aux_bc_lower[0] = pyclaw.BC.extrap
        solver.aux_bc_upper[0] = pyclaw.BC.extrap
        solver.aux_bc_lower[1] = pyclaw.BC.extrap
        solver.aux_bc_upper[1] = pyclaw.BC.extrap

        solver.fwave = True
        solver.step_source = forcing.wind_forcing_step
        solver.source_split = 2

        # Define Dimensions
        x = pyclaw.Dimension(*self.x_domain, self.nx, name="x")
        y = pyclaw.Dimension(*self.y_domain, self.ny, name="y")
        domain = pyclaw.Domain([x, y])

        # Define State
        state = pyclaw.State(domain, num_eqn, num_aux=1)
        print(type(state))
        state.problem_data["grav"] = self.gravity

        # Initialize Data
        self._initialize_solution_state(state)

        # Setup Controller
        claw = pyclaw.Controller()
        claw.tfinal = self.t_final
        claw.solution = pyclaw.Solution(state, domain, outdir="_output")
        claw.solver = solver

        # Calculate output times based on dt
        if self.multiple_output_times:
            claw.num_output_times = int(self.t_final / self.dt)
        else:
            claw.num_output_times = 1

        claw.keep_copy = True
        self.claw = claw

    def solve(self):
        """Run the simulation."""
        if self.claw is None:
            self.setup_solver()
        return self.claw.run()


if __name__ == "__main__":
    # Example usage
    solver = sweSolver()
    solver.set_time_parameters(t_final=10.0, dt=0.5)
    solver.set_domain(lon_range=(-10, 10), lat_range=(-10, 10), nx=100, ny=100)

    # Example bathymetry: flat sea floor at -10m
    bathymetry = -10 * np.ones((solver.ny, solver.nx))
    solver.set_bathymetry(bathymetry)

    # Example initial condition: Gaussian hump
    x = np.linspace(-10, 10, solver.nx)
    y = np.linspace(-10, 10, solver.ny)
    X, Y = np.meshgrid(x, y)
    initial_condition = 5.0 * np.exp(-0.1 * (X**2 + Y**2))
    solver.set_initial_condition(initial_condition)

    solver.set_boundary_conditions(
        lower=[pyclaw.BC.wall, pyclaw.BC.wall], upper=[pyclaw.BC.wall, pyclaw.BC.wall]
    )

    claw_controller = solver.setup_solver()
    claw_controller.run()
