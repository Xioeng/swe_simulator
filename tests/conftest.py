"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

from swe_simulator.config import SimulationConfig
from swe_simulator.coordinate_mapper import GeographicCoordinateMapper
from swe_simulator.forcing import WindForcing


@pytest.fixture
def basic_config():
    """Basic valid configuration for testing."""
    return SimulationConfig(
        lon_range=(-1.0, 1.0),
        lat_range=(-1.0, 1.0),
        nx=50,
        ny=50,
        t_final=10.0,
        dt=0.1,
    )


@pytest.fixture
def sample_bathymetry():
    """Sample flat bathymetry array (-10m depth)."""
    return -10.0 * np.ones((50, 50))


@pytest.fixture
def sample_initial_condition():
    """Sample Gaussian hump initial condition."""
    nx, ny = 50, 50
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    h = 2.0 * np.exp(-0.01 * (X**2 + Y**2))
    hu = np.zeros_like(h)
    hv = np.zeros_like(h)
    return np.stack([h, hu, hv])


@pytest.fixture
def coordinate_mapper():
    """Create a geographic coordinate mapper."""
    return GeographicCoordinateMapper(lon0=0.0, lat0=0.0)


@pytest.fixture
def wind_forcing():
    """Create a wind forcing object."""
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(-1.0, 1.0, 10),
        np.linspace(-1.0, 1.0, 10),
        indexing="ij",
    )
    return WindForcing(mesgrid_domain=(lon_grid, lat_grid))


@pytest.fixture
def simple_domain():
    max_lon = 0.0005  # half-width in longitude (degrees)
    max_lat = 0.0005  # half-width in latitude (degrees)

    lon_range = (-max_lon, max_lon)
    lat_range = (-max_lat, max_lat)
    return np.meshgrid(
        np.linspace(lon_range[0], lon_range[1], 40),
        np.linspace(lat_range[0], lat_range[1], 40),
        indexing="ij",
    )
