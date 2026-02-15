#!/usr/bin/env python
# encoding: utf-8

import os
from typing import Optional, Tuple

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
from .logging_config import get_logger
from .result import SWEResult
from .utils.grid import generate_cell_centers

logger = get_logger(__name__)


class SWESolver:
    """
    Solver class for 2D shallow water equations using PyClaw.

    Parameters
    ----------
    multiple_output_times : bool, default=True
        Whether to output at multiple time steps
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        # Simulation parameters
        self.config = config or SimulationConfig()
        self.wind_forcing: WindForcing = WindForcing()
        # Arrays
        self.bathymetry_array: np.ndarray = np.zeros((self.config.ny, self.config.nx))
        self.initial_condition_array: np.ndarray = np.zeros(
            (3, self.config.ny, self.config.nx)
        )

        # MPI
        self.comm: MPI.Comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()

        if self.config.lon_range is not None and self.config.lat_range is not None:
            logger.info("Setting domain in init")
            self.set_domain(
                self.config.lon_range,
                self.config.lat_range,
                self.config.nx,
                self.config.ny,
            )

    def set_time_parameters(self, t_final: float, dt: float) -> None:
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
    ) -> None:
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

    def _check_arrays_sanity_set(
        self, array: np.ndarray, expected_shape: Tuple[int, ...], name: str
    ) -> list:
        errors = []
        if array is None:
            errors.append(f"{name} has not been set.")
            return errors
        """Check if a required array has the correct shape."""
        if array.shape != expected_shape:
            errors.append(
                f"{name} has incorrect shape. Expected {expected_shape}, got {array.shape}."
            )
        return errors

    def set_bathymetry(self, bathymetry_array: np.ndarray) -> None:
        """
        Set the bathymetry for the domain.

        Parameters
        ----------
        bathymetry_array : np.ndarray
            Array of shape (ny, nx) with bathymetry values
        """
        self.bathymetry_array = bathymetry_array

    def set_initial_condition(self, initial_condition: np.ndarray) -> None:
        """
        Set the initial condition.

        Parameters
        ----------
        initial_condition : np.ndarray
            Array of shape (3, ny, nx) with [h, hu, hv]
        """
        self.initial_condition_array = initial_condition

    def set_boundary_conditions(
        self, lower: Tuple[int, int], upper: Tuple[int, int]
    ) -> None:
        """
        Set boundary conditions.

        Parameters
        ----------
        lower : Tuple[int, int]
            BCs for x-lower and y-lower [x_lo, y_lo]
        upper : Tuple[int, int]
            BCs for x-upper and y-upper [x_hi, y_hi]
        """
        self.config.bc_lower = lower
        self.config.bc_upper = upper

    def set_wind_forcing(
        self,
        u_wind: float = 0.0,
        v_wind: float = 0.0,
        c_d: float = 1.3e-3,
    ) -> None:
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

    def _validate_swe_configuration(self) -> None:
        """Validate that all required configuration has been set."""

        errors = []
        for array, shape, name in [
            (
                self.bathymetry_array,
                (self.config.ny, self.config.nx),
                "Bathymetry array",
            ),
            (
                self.initial_condition_array,
                (3, self.config.ny, self.config.nx),
                "Initial condition array",
            ),
        ]:
            errors.extend(self._check_arrays_sanity_set(array, shape, name))

        if errors:
            logger.error("SWE configuration errors found:\n" + "\n".join(errors))
            raise ValueError("SWE configuration errors found:\n" + "\n".join(errors))

    def _initialize_solution_state(self, state) -> None:
        """Initialize the solution state with initial conditions."""
        # Ensure output directory exists
        if self.config.output_dir and self.rank == 0:
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)

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
        self._validate_swe_configuration()

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
            logger.info("Applying wind forcing to solver")
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
        claw.solution = pyclaw.Solution(state, domain)
        if self.config.output_dir:
            claw.outdir = self.config.output_dir
        else:
            claw.output_format = None
        claw.solver = solver

        # Output times
        if self.config.multiple_output_times:
            claw.num_output_times = int(self.config.t_final / self.config.dt)
        else:
            claw.num_output_times = 1

        claw.keep_copy = True
        claw.verbosity = 3
        self.claw = claw

        return claw

    def solve(self) -> SWEResult:
        """Run the simulation."""
        if self.claw is None:
            self.setup_solver()
        self.claw.run()

        solutions = np.stack([frame.q for frame in self.claw.frames])
        result = SWEResult(
            meshgrid_coord=(self.X_coord, self.Y_coord),
            meshgrid_metric=(self.X, self.Y),
            solution=solutions,
            bathymetry=self.bathymetry_array,
            initial_condition=self.initial_condition_array,
            wind_forcing=self.wind_forcing.get_wind(),
            config=self.config,
        )
        if self.config.output_dir is not None and self.rank == 0:
            result.save(os.path.join(self.config.output_dir, "result.pkl"))
        return result
