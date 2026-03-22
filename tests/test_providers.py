"""Tests for data providers."""

import numpy as np
import pytest

from swe_simulator.providers import (
    BathymetryProvider,
    ConstantWind,
    FlatBathymetry,
    GaussianHumpInitialCondition,
    InitialConditionProvider,
    SlopingBathymetry,
    # TimeVaryingWind,
    WindProvider,
)


def _build_lon_lat_grid(basic_config):
    lon = np.linspace(
        basic_config.lon_range[0], basic_config.lon_range[1], basic_config.nx
    )
    lat = np.linspace(
        basic_config.lat_range[0], basic_config.lat_range[1], basic_config.ny
    )
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")
    return lon_grid, lat_grid


class TestInitialConditionProvider:
    """Test initial condition provider interface and implementations."""

    def test_abstract_class_cannot_be_instantiated(self):
        """InitialConditionProvider should not be instantiable."""
        with pytest.raises(TypeError):
            InitialConditionProvider()

    def test_gaussian_hump_shape(self, basic_config):
        """Gaussian hump should return correct shape."""
        lon_grid, lat_grid = _build_lon_lat_grid(basic_config)
        ic = GaussianHumpInitialCondition()
        result = ic.get_initial_condition(lon_grid, lat_grid)

        assert result.shape == (3, basic_config.ny, basic_config.nx)

    def test_gaussian_hump_values(self, basic_config):
        """Gaussian hump values should be in reasonable ranges."""
        lon_grid, lat_grid = _build_lon_lat_grid(basic_config)
        ic = GaussianHumpInitialCondition(
            bias=0.0,
            height=2.0,
            width=8.0,
            center=(0.0, 0.0),
        )
        result = ic.get_initial_condition(lon_grid, lat_grid)

        h, hu, hv = result[0], result[1], result[2]

        # Height should be positive
        assert np.all(h >= 0)
        # Max height should be bounded by the Gaussian peak
        assert np.max(h) <= 2.0 + 1e-12
        # Peak should occur near the prescribed center
        peak_index = np.unravel_index(np.argmax(h), h.shape)
        assert np.isclose(lon_grid[peak_index], 0.0, atol=0.05)
        assert np.isclose(lat_grid[peak_index], 0.0, atol=0.05)
        # Momentum should be zero (no initial flow)
        assert np.allclose(hu, 0.0)
        assert np.allclose(hv, 0.0)


class TestWindProvider:
    """Test wind provider interface and implementations."""

    def test_abstract_class_cannot_be_instantiated(self):
        """WindProvider should not be instantiable."""
        with pytest.raises(TypeError):
            WindProvider()

    def test_constant_wind(self):
        """Constant wind should return same values."""
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(-1.0, 1.0, 5),
            np.linspace(-1.0, 1.0, 4),
            indexing="xy",
        )
        wind = ConstantWind(u_wind=5.0, v_wind=2.0)

        u1, v1 = wind.get_wind(lon_grid, lat_grid, time=0.0)
        u2, v2 = wind.get_wind(lon_grid, lat_grid, time=10.0)

        assert u1.shape == lon_grid.shape
        assert v1.shape == lat_grid.shape
        assert np.allclose(u1, 5.0)
        assert np.allclose(v1, 2.0)
        assert np.allclose(u2, 5.0)
        assert np.allclose(v2, 2.0)


class TestBathymetryProvider:
    """Test bathymetry provider interface and implementations."""

    def test_abstract_class_cannot_be_instantiated(self):
        """BathymetryProvider should not be instantiable."""
        with pytest.raises(TypeError):
            BathymetryProvider()

    def test_flat_bathymetry_shape(self, basic_config):
        """Flat bathymetry should return correct shape."""
        lon_grid, lat_grid = _build_lon_lat_grid(basic_config)
        bathy = FlatBathymetry(depth=-10.0)
        result = bathy.get_bathymetry(lon_grid, lat_grid)

        assert result.shape == (basic_config.ny, basic_config.nx)

    def test_flat_bathymetry_values(self, basic_config):
        """Flat bathymetry should have uniform depth."""
        lon_grid, lat_grid = _build_lon_lat_grid(basic_config)
        depth = -15.0
        bathy = FlatBathymetry(depth=depth)
        result = bathy.get_bathymetry(lon_grid, lat_grid)

        assert np.allclose(result, depth)

    def test_sloping_bathymetry(self, basic_config):
        """Sloping bathymetry should vary with y-coordinate."""
        lon_grid, lat_grid = _build_lon_lat_grid(basic_config)
        bathy = SlopingBathymetry(depth_min=-5.0, depth_max=-20.0)
        result = bathy.get_bathymetry(lon_grid, lat_grid)

        assert result.shape == (basic_config.ny, basic_config.nx)
        # Depth should increase from min to max
        assert np.isclose(np.min(result), -20.0)
        assert np.isclose(np.max(result), -5.0)
        # Sloping direction should be monotonic along latitude axis
        row_means = result.mean(axis=1)
        assert np.all(np.diff(row_means) <= 1e-12)
