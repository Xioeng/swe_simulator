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

from .config import SimulationConfig
from .coordinate_mapper import GeographicCoordinateMapper
from .forcing import WindForcing
from .utils.grid import generate_cell_centers


class SWESolver:
    """
    Solver class for 2D shallow water equations using PyClaw.

    Parameters
    ----------
    multiple_output_times : bool, default=True
        Whether to output at multiple time steps
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        # Simulation parameters
        self.config = config or SimulationConfig()
        self.wind_forcing: WindForcing = WindForcing()
        # Arrays
        self.bathymetry_array: Optional[np.ndarray] = None
        self.initial_condition_array: Optional[np.ndarray] = None

        # MPI
        self.comm: MPI.Comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()

        self.mapper: Optional[GeographicCoordinateMapper] = None
        # Internal state
        self.claw: Optional[pyclaw.Controller] = None

        if self.config.lon_range is not None and self.config.lat_range is not None:
            print("setting domain in init")
            self.set_domain(
                self.config.lon_range,
                self.config.lat_range,
                self.config.nx,
                self.config.ny,
            )

    def set_time_parameters(self, t_final: float, dt: float):
        """
        Set the time parameters for the simulation.

        Parameters
        ----------
        t_final : float
            Final simulation time
        dt : float
            Time step size
        """
        self.config.t_final = t_final
        self.config.dt = dt

    def set_domain(
        self,
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        nx: int,
        ny: int,
    ):
        """
        Set up the grid domain based on longitude and latitude ranges.

        Parameters
        ----------
        lon_range : Tuple[float, float]
            (min_lon, max_lon) in degrees
        lat_range : Tuple[float, float]
            (min_lat, max_lat) in degrees
        nx, ny : int
            Number of cells in each direction
        """
        self.config.nx = nx
        self.config.ny = ny

        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        self.config.lon_range = lon_range
        self.config.lat_range = lat_range

        self.mapper = GeographicCoordinateMapper(
            (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
        )
        mins = self.mapper.coord_to_metric(lon_min, lat_min)
        maxs = self.mapper.coord_to_metric(lon_max, lat_max)

        self.x_domain = [float(mins[0]), float(maxs[0])]
        self.y_domain = [float(mins[1]), float(maxs[1])]
        self.X, self.Y = generate_cell_centers(
            self.x_domain[0],
            self.x_domain[1],
            self.y_domain[0],
            self.y_domain[1],
            self.config.nx,
            self.config.ny,
        )
        self.X_coord, self.Y_coord = self.mapper.metric_to_coord(self.X, self.Y)

    def set_bathymetry(self, bathymetry_array: np.ndarray):
        """
        Set the bathymetry for the domain.

        Parameters
        ----------
        bathymetry_array : np.ndarray
            Array of shape (ny, nx) with bathymetry values
        """
        assert bathymetry_array.shape == (
            self.config.ny,
            self.config.nx,
        ), "Bathymetry array must match grid dimensions"
        self.bathymetry_array = bathymetry_array

    def set_initial_condition(self, initial_condition: np.ndarray):
        """
        Set the initial condition.

        Parameters
        ----------
        initial_condition : np.ndarray
            Array of shape (3, ny, nx) with [h, hu, hv]
        """
        assert initial_condition.shape == (
            3,
            self.config.ny,
            self.config.nx,
        ), "Initial condition array must match grid dimensions"
        self.initial_condition_array = initial_condition

    def set_boundary_conditions(self, lower: List[int], upper: List[int]):
        """
        Set boundary conditions.

        Parameters
        ----------
        lower : List[int]
            BCs for x-lower and y-lower [x_lo, y_lo]
        upper : List[int]
            BCs for x-upper and y-upper [x_hi, y_hi]
        """
        self.config.bc_lower = lower
        self.config.bc_upper = upper

    def set_wind_forcing(
        self,
        u_wind: float = 0.0,
        v_wind: float = 0.0,
        c_d: float = 1.3e-3,
    ):
        """
        Set wind forcing parameters.

        Parameters
        ----------
        u_wind : float, default=0.0
            Wind velocity in x-direction (m/s)
        v_wind : float, default=0.0
            Wind velocity in y-direction (m/s)
        c_d : float, default=1.3e-3
            Drag coefficient
        """
        self.wind_forcing = WindForcing(u_wind=u_wind, v_wind=v_wind, c_d=c_d)

    def _validate_configuration(self):
        """Validate that all required configuration has been set."""
        self.config.validate()
        self.set_domain(
            self.config.lon_range,
            self.config.lat_range,
            self.config.nx,
            self.config.ny,
        )

        if self.bathymetry_array is None:
            raise ValueError("Bathymetry array has not been set.")
        if self.initial_condition_array is None:
            raise ValueError("Initial condition array has not been set.")
        if self.mapper is None:
            raise ValueError("Domain has not been set. Call set_domain() first.")

    def _initialize_solution_state(self, state):
        """Initialize the solution state with initial conditions."""
        # Ensure output directory exists
        if not os.path.exists(self.config.output_dir) and self.rank == 0:
            os.makedirs(self.config.output_dir)

        # Save grid metadata
        if self.rank == 0:
            np.save(
                f"{self.config.output_dir}/coord_meshgrid.npy",
                np.stack((self.X_coord, self.Y_coord), axis=0),
            )
            np.save(f"{self.config.output_dir}/bathymetry.npy", self.bathymetry_array)

        # Set state
        h = np.maximum(0.0, self.initial_condition_array[0] - self.bathymetry_array)
        state.q[depth, :, :] = h
        state.q[x_momentum, :, :] = self.initial_condition_array[1]
        state.q[y_momentum, :, :] = self.initial_condition_array[2]
        state.aux[:, :, :] = self.bathymetry_array

    def setup_solver(self) -> pyclaw.Controller:
        """
        Construct the PyClaw solver, domain, and controller.

        Returns
        -------
        pyclaw.Controller
            Configured PyClaw controller
        """
        self._validate_configuration()

        # Create solver
        rs = riemann.sw_aug_2D
        solver = pyclaw.ClawSolver2D(rs)

        # Boundary conditions
        solver.bc_lower[0] = self.config.bc_lower[0]
        solver.bc_upper[0] = self.config.bc_upper[0]
        solver.bc_lower[1] = self.config.bc_lower[1]
        solver.bc_upper[1] = self.config.bc_upper[1]

        # Aux BCs
        solver.aux_bc_lower[0] = pyclaw.BC.extrap
        solver.aux_bc_upper[0] = pyclaw.BC.extrap
        solver.aux_bc_lower[1] = pyclaw.BC.extrap
        solver.aux_bc_upper[1] = pyclaw.BC.extrap

        solver.fwave = True

        # Set wind forcing if configured
        if self.wind_forcing is not None:
            solver.step_source = self.wind_forcing  # Use the callable instance
            solver.source_split = 2

        # Define domain
        x = pyclaw.Dimension(*self.x_domain, self.config.nx, name="x")
        y = pyclaw.Dimension(*self.y_domain, self.config.ny, name="y")
        domain = pyclaw.Domain([x, y])

        # Define state
        state = pyclaw.State(domain, num_eqn, num_aux=1)
        state.problem_data["grav"] = self.config.gravity

        # Initialize data
        self._initialize_solution_state(state)

        # Setup controller
        claw = pyclaw.Controller()
        claw.tfinal = self.config.t_final
        claw.solution = pyclaw.Solution(state, domain, outdir=self.config.output_dir)
        claw.solver = solver

        # Output times
        if self.config.multiple_output_times:
            claw.num_output_times = int(self.config.t_final / self.config.dt)
        else:
            claw.num_output_times = 1

        claw.keep_copy = True
        self.claw = claw
        return claw

    def solve(self):
        """Run the simulation."""
        if self.claw is None:
            self.setup_solver()
        return self.claw.run()


if __name__ == "__main__":
    # Example usage
    solver = SWESolver()
    solver.set_time_parameters(t_final=10.0, dt=0.5)
    solver.set_domain(lon_range=(-10, 10), lat_range=(-10, 10), nx=100, ny=100)

    # Bathymetry: flat sea floor at -10m
    bathymetry = -10 * np.ones((solver.ny, solver.nx))
    solver.set_bathymetry(bathymetry)

    # Initial condition: Gaussian hump
    x = np.linspace(-10, 10, solver.nx)
    y = np.linspace(-10, 10, solver.ny)
    X, Y = np.meshgrid(x, y)
    h_init = 5.0 * np.exp(-0.1 * (X**2 + Y**2))
    initial_condition = np.array([h_init, np.zeros_like(h_init), np.zeros_like(h_init)])
    solver.set_initial_condition(initial_condition)

    solver.set_boundary_conditions(
        lower=[pyclaw.BC.wall, pyclaw.BC.wall], upper=[pyclaw.BC.wall, pyclaw.BC.wall]
    )

    # Set wind forcing
    solver.set_wind_forcing(u_wind=10.0, v_wind=5.0)
    # Or use meteorological convention:
    # solver.set_wind_from_speed_direction(speed=15.0, direction_deg=45)

    solver.solve()
