#!/usr/bin/env python
# encoding: utf-8

import os
from typing import cast

import clawpack.pyclaw as pyclaw
import numpy as np
import numpy.typing as npt
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
from .providers import (
    BathymetryProvider,
    ConstantWind,
    InitialConditionProvider,
    WindProvider,
)
from .result import SWEResult
from .utils.grid import generate_cell_centers

logger = get_logger(__name__)


class SWESolver:
    """
    Solver class for 2D shallow water equations using PyClaw.

    Parameters
    ----------
    config : SimulationConfig, optional
        Simulation configuration
    ic_provider : InitialConditionProvider, optional
        Provider for initial conditions. Defaults to FlatInitialCondition.
    wind_provider : WindProvider, optional
        Provider for wind forcing. Defaults to ConstantWind().
    bathymetry_provider : BathymetryProvider, optional
        Provider for bathymetry. Defaults to FlatBathymetry().
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        ic_provider: InitialConditionProvider | None = None,
        wind_provider: WindProvider = ConstantWind(),
        bathymetry_provider: BathymetryProvider | None = None,
    ) -> None:
        # Simulation parameters
        self.config = config or SimulationConfig()

        # Data providers with sensible defaults
        self.ic_provider = ic_provider
        self.wind_provider = wind_provider
        self.bathymetry_provider = bathymetry_provider

        # Arrays (will be populated from providers or manual setters)
        self.bathymetry_array: npt.NDArray[np.float64] = np.zeros(
            (self.config.ny, self.config.nx)
        )
        self.initial_condition_array: npt.NDArray[np.float64] = np.zeros(
            (3, self.config.ny, self.config.nx)
        )

        # MPI
        self.comm: MPI.Comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()

        # PyClaw objects
        self.claw: pyclaw.Controller

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
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        nx: int,
        ny: int,
    ) -> None:
        """
        Set up the grid domain based on longitude and latitude ranges.

        Parameters
        ----------
        lon_range : tuple[float, float]
            (min_lon, max_lon) in degrees
        lat_range : tuple[float, float]
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
        lon_lat = self.mapper.metric_to_coord(self.X, self.Y)
        self.X_coord, self.Y_coord = cast(
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
            lon_lat,
        )

    @staticmethod
    def _check_arrays_sanity_set(
        array: npt.NDArray[np.float64], expected_shape: tuple[int, ...], name: str
    ) -> list:
        errors = []
        if array is None:
            errors.append(f"{name} has not been set.")
            return errors
        """Check if a required array has the correct shape."""
        if array.shape != expected_shape:
            errors.append(
                f"{name} has incorrect shape. "
                f"Expected {expected_shape}, got {array.shape}."
            )
        return errors

    def initialize_data_from_providers(self) -> None:
        """
        Generate arrays from configured providers.

        This method populates bathymetry and initial condition arrays
        using the current providers. Requires domain to be set first.
        """
        if not hasattr(self, "X_coord") or not hasattr(self, "Y_coord"):
            raise RuntimeError(
                "Domain must be set before initializing arrays from providers. "
                "Call set_domain() first."
            )

        logger.info("Initializing arrays from providers")
        if self.bathymetry_provider is not None:
            self.bathymetry_array = self.bathymetry_provider.get_bathymetry(
                self.X_coord, self.Y_coord
            )
        if self.ic_provider is not None:
            self.initial_condition_array = self.ic_provider.get_initial_condition(
                self.X_coord, self.Y_coord
            )

        self.wind_forcing = WindForcing(
            mesgrid_domain=(self.X_coord, self.Y_coord),
            wind_provider=self.wind_provider,
        )

    def set_bathymetry(self, bathymetry_array: npt.NDArray[np.float64]) -> None:
        """
        Set the bathymetry for the domain.

        Parameters
        ----------
        bathymetry_array : npt.NDArray[np.float64]
            Array of shape (ny, nx) with bathymetry values

        Notes
        -----
        Setting bathymetry directly will override any bathymetry provider.
        """
        self.bathymetry_array = bathymetry_array
        self.bathymetry_provider = None

    def set_initial_condition(self, initial_condition: npt.NDArray[np.float64]) -> None:
        """
        Set the initial condition.

        Parameters
        ----------
        initial_condition : npt.NDArray[np.float64]
            Array of shape (3, ny, nx) with [h, hu, hv]

        Notes
        -----
        Setting initial condition directly will override any IC provider.
        """
        self.initial_condition_array = initial_condition
        self.ic_provider = None

    def set_boundary_conditions(
        self, lower: tuple[int, int], upper: tuple[int, int]
    ) -> None:
        """
        Set boundary conditions.

        Parameters
        ----------
        lower : tuple[int, int]
            BCs for x-lower and y-lower [x_lo, y_lo]
        upper : tuple[int, int]
            BCs for x-upper and y-upper [x_hi, y_hi]
        """
        self.config.bc_lower = lower
        self.config.bc_upper = upper

    def set_constant_wind_forcing(
        self,
        u_wind: float = 0.0,
        v_wind: float = 0.0,
    ) -> None:
        """
        Set wind forcing parameters (legacy method).

        Parameters
        ----------
        u_wind : float, default=0.0
            Wind velocity in x-direction (m/s)
        v_wind : float, default=0.0
            Wind velocity in y-direction (m/s)

        Notes
        -----
        This method is deprecated. Use set_wind_provider() instead.
        """
        # So far, construct a constant wind forcing
        # To Do: extend this to support time-varying or spatially-varying wind

        # Also update provider for consistency
        self.wind_provider = ConstantWind(u_wind=u_wind, v_wind=v_wind)

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
        h = self.initial_condition_array[0] - self.bathymetry_array
        state.q[depth, :, :] = h
        state.q[x_momentum, :, :] = self.initial_condition_array[1]
        state.q[y_momentum, :, :] = self.initial_condition_array[2]
        state.aux[:, :, :] = self.bathymetry_array

    def _create_pyclaw_solver(self) -> pyclaw.ClawSolver2D:
        """Create and configure the low-level PyClaw solver."""
        rs = riemann.sw_aug_2D
        solver = pyclaw.ClawSolver2D(rs)
        solver.fwave = True
        solver.verbosity = 0
        return solver

    def _configure_boundary_conditions(self, solver) -> None:
        """Apply configured physical boundary conditions to the solver."""
        solver.bc_lower[0] = self.config.bc_lower[0]
        solver.bc_upper[0] = self.config.bc_upper[0]
        solver.bc_lower[1] = self.config.bc_lower[1]
        solver.bc_upper[1] = self.config.bc_upper[1]

    def _configure_aux_boundary_conditions(self, solver) -> None:
        """Apply default auxiliary boundary conditions to the solver."""
        solver.aux_bc_lower[0] = pyclaw.BC.extrap
        solver.aux_bc_upper[0] = pyclaw.BC.extrap
        solver.aux_bc_lower[1] = pyclaw.BC.extrap
        solver.aux_bc_upper[1] = pyclaw.BC.extrap

    def _configure_source_terms(self, solver) -> None:
        """Attach source terms (wind forcing) to the solver when available."""
        if self.wind_forcing is not None:
            logger.info("Applying wind forcing to solver")
            solver.step_source = self.wind_forcing
            solver.source_split = 2

    def _create_domain(self) -> pyclaw.Domain:
        """Create the PyClaw computational domain from configured metric bounds."""
        x = pyclaw.Dimension(*self.x_domain, self.config.nx, name="x")
        y = pyclaw.Dimension(*self.y_domain, self.config.ny, name="y")
        return pyclaw.Domain([x, y])

    def _create_state(self, domain) -> pyclaw.State:
        """Create and initialize the PyClaw state object."""
        state = pyclaw.State(domain, num_eqn, num_aux=1)
        state.problem_data["grav"] = self.config.gravity
        self._initialize_solution_state(state)
        return state

    def _create_controller(self, solver, state, domain) -> pyclaw.Controller:
        """Create and configure the high-level PyClaw controller."""
        claw = pyclaw.Controller()
        claw.logger = logger
        claw.tfinal = self.config.t_final
        claw.solution = pyclaw.Solution(state, domain)

        if self.config.output_dir:
            claw.outdir = self.config.output_dir
        else:
            claw.output_format = None

        claw.solver = solver

        if self.config.multiple_output_times:
            claw.num_output_times = int(self.config.t_final / self.config.dt)
        else:
            claw.num_output_times = 1

        claw.keep_copy = True
        # claw.verbosity = 0
        return claw

    def setup_solver(self) -> None:
        """
        Construct the PyClaw solver, domain, and controller.

        Returns
        -------
        pyclaw.Controller
            Configured PyClaw controller
        """
        # Generate arrays from providers if not already set manually
        self.initialize_data_from_providers()

        self._validate_swe_configuration()

        solver = self._create_pyclaw_solver()
        self._configure_boundary_conditions(solver)
        self._configure_aux_boundary_conditions(solver)
        self._configure_source_terms(solver)

        domain = self._create_domain()
        state = self._create_state(domain)
        claw = self._create_controller(solver, state, domain)

        self.claw = claw

    def solve(self) -> SWEResult:
        """Run the simulation."""

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
