"""Tests for solver module."""

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("clawpack.petclaw")

from tidalflow.providers import BathymetryProvider, InitialConditionProvider
from tidalflow.solver import SWESolver


class DummyBathymetryProvider(BathymetryProvider):
    """Simple bathymetry provider for tests."""

    def __init__(self, depth: float = -7.0):
        self.depth = depth

    def get_bathymetry(
        self, lon: npt.NDArray[np.float64], lat: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return self.depth * np.ones_like(lon)


class DummyInitialConditionProvider(InitialConditionProvider):
    """Simple initial condition provider for tests."""

    def __init__(self, depth: float = 1.5):
        self.depth = depth

    def get_initial_condition(
        self, lon: npt.NDArray[np.float64], lat: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        h = self.depth * np.ones_like(lon)
        hu = np.zeros_like(lon)
        hv = np.zeros_like(lon)
        return np.stack([h, hu, hv])


def test_init_sets_domain_from_config(basic_config):
    """Solver should initialize domain arrays when ranges are configured."""
    solver = SWESolver(config=basic_config)

    assert solver.config.lon_range == basic_config.lon_range
    assert solver.config.lat_range == basic_config.lat_range
    assert solver.X.shape == (basic_config.nx, basic_config.ny)
    assert solver.Y.shape == (basic_config.nx, basic_config.ny)
    assert solver.X_coord.shape == (basic_config.nx, basic_config.ny)
    assert solver.Y_coord.shape == (basic_config.nx, basic_config.ny)


def test_set_time_parameters_updates_config(basic_config):
    """Time parameters should be writable via setter."""
    solver = SWESolver(config=basic_config)

    solver.set_time_parameters(t_final=25.0, dt=0.25)

    assert solver.config.t_final == 25.0
    assert solver.config.dt == 0.25


def test_set_domain_updates_grid_and_config(basic_config):
    """Domain setter should update bounds, sizes, and generated grids."""
    solver = SWESolver(config=basic_config)

    lon_range = (-0.2, 0.4)
    lat_range = (-0.1, 0.3)
    nx, ny = 12, 8
    solver.set_domain(lon_range=lon_range, lat_range=lat_range, nx=nx, ny=ny)

    assert solver.config.lon_range == lon_range
    assert solver.config.lat_range == lat_range
    assert solver.config.nx == nx
    assert solver.config.ny == ny
    assert solver.X.shape == (nx, ny)
    assert solver.Y.shape == (nx, ny)
    assert solver.X_coord.shape == (nx, ny)
    assert solver.Y_coord.shape == (nx, ny)


def test_set_bathymetry_disables_provider(basic_config):
    """Manual bathymetry should clear configured bathymetry provider."""
    solver = SWESolver(config=basic_config)
    solver.bathymetry_provider = DummyBathymetryProvider(depth=-11.0)
    bathy = -10.0 * np.ones((basic_config.ny, basic_config.nx))

    solver.set_bathymetry(bathy)

    np.testing.assert_allclose(solver.bathymetry_array, bathy)
    assert solver.bathymetry_provider is None


def test_set_initial_condition_disables_provider(basic_config):
    """Manual initial condition should clear configured IC provider."""
    solver = SWESolver(config=basic_config)
    solver.ic_provider = DummyInitialConditionProvider(depth=2.0)
    init = np.zeros((3, basic_config.ny, basic_config.nx))

    solver.set_initial_condition(init)

    np.testing.assert_allclose(solver.initial_condition_array, init)
    assert solver.ic_provider is None


def test_initialize_data_from_providers_requires_domain(basic_config):
    """Provider initialization should fail when domain arrays are missing."""
    solver = SWESolver(config=basic_config)
    del solver.X_coord
    del solver.Y_coord

    with pytest.raises(RuntimeError, match="Domain must be set"):
        solver.initialize_data_from_providers()


def test_initialize_data_from_providers_populates_arrays(basic_config):
    """Provider initialization should populate bathy, IC, and wind forcing."""
    solver = SWESolver(
        config=basic_config,
        ic_provider=DummyInitialConditionProvider(depth=1.25),
        bathymetry_provider=DummyBathymetryProvider(depth=-8.0),
    )

    solver.initialize_data_from_providers()

    assert solver.bathymetry_array.shape == (basic_config.nx, basic_config.ny)
    assert solver.initial_condition_array.shape == (
        3,
        basic_config.nx,
        basic_config.ny,
    )
    np.testing.assert_allclose(solver.bathymetry_array, -8.0)
    np.testing.assert_allclose(solver.initial_condition_array[0], 1.25)
    assert solver.wind_forcing is not None


def test_check_arrays_sanity_set_detects_wrong_shape(basic_config):
    """Shape checks should return a useful error message."""
    solver = SWESolver(config=basic_config)
    wrong_shape = np.zeros((basic_config.nx, basic_config.ny))

    errors = solver._check_arrays_sanity_set(
        wrong_shape,
        expected_shape=(basic_config.ny, basic_config.nx),
        name="Bathymetry array",
    )
    assert len(errors) == 1
    assert "incorrect shape" in errors[0]


def test_validate_swe_configuration_raises_for_bad_shapes(basic_config):
    """Configuration validation should raise when array dimensions are invalid."""
    solver = SWESolver(config=basic_config)
    solver.bathymetry_array = np.zeros((basic_config.nx, basic_config.ny))
    solver.initial_condition_array = np.zeros((3, basic_config.nx, basic_config.ny))

    with pytest.raises(ValueError, match="SWE configuration errors found"):
        solver._validate_swe_configuration()
