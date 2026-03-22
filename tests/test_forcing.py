"""Tests for wind forcing module."""

import numpy as np
import pytest

from swe_simulator.forcing import WindForcing
from swe_simulator.providers import ConstantWind


def _build_test_grid(nx: int = 10, ny: int = 10) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    return np.meshgrid(x, y, indexing="xy")


def _make_wind_forcing(
    u_wind: float = 5.0,
    v_wind: float = 2.0,
    c_d: float = 1.3e-3,
    rho_air: float = 1.2,
    rho_water: float = 1000.0,
) -> WindForcing:
    lon_grid, lat_grid = _build_test_grid()
    provider = ConstantWind(u_wind=u_wind, v_wind=v_wind)
    return WindForcing(
        mesgrid_domain=(lon_grid, lat_grid),
        c_d=c_d,
        rho_air=rho_air,
        rho_water=rho_water,
        wind_provider=provider,
    )


class TestWindForcing:
    """Test wind forcing initialization and calculations."""

    def test_wind_forcing_initialization(self):
        """Test basic initialization."""
        wind = _make_wind_forcing(u_wind=5.0, v_wind=2.0)
        u, v = wind.get_wind()
        assert np.allclose(u, 5.0)
        assert np.allclose(v, 2.0)

    def test_wind_forcing_defaults(self):
        """Test default values."""
        wind = _make_wind_forcing(u_wind=0.0, v_wind=0.0)
        u, v = wind.get_wind()
        assert np.allclose(u, 0.0)
        assert np.allclose(v, 0.0)
        assert wind.c_d > 0  # drag coefficient should be positive
        assert wind.rho_air > 0
        assert wind.rho_water > 0

    def test_wind_forcing_custom_density(self):
        """Test custom density values."""
        wind = _make_wind_forcing(rho_air=1.3, rho_water=1025.0)
        assert wind.rho_air == 1.3
        assert wind.rho_water == 1025.0

    def test_get_wind_velocity(self):
        """Test getting wind velocity."""
        wind_forcing = _make_wind_forcing(u_wind=5.0, v_wind=0.0)
        u, v = wind_forcing.get_wind()
        assert u.shape == v.shape
        assert np.allclose(u, 5.0)
        assert np.allclose(v, 0.0)

    def test_set_drag_coefficient(self):
        """Test setting drag coefficient."""
        wind_forcing = _make_wind_forcing()
        wind_forcing.set_drag_coefficient(1.5e-3)
        assert wind_forcing.get_drag_coefficient() == 1.5e-3

    def test_compute_velocities_flat_water(self):
        """Test velocity computation for uniform water."""
        h = np.ones((10, 10)) * 5.0  # 5m depth everywhere
        hu = np.ones((10, 10)) * 2.0  # uniform x-momentum
        hv = np.ones((10, 10)) * 1.0  # uniform y-momentum

        u, v, mask = WindForcing.compute_velocities(h, hu, hv)

        # u = hu/h = 2/5 = 0.4
        np.testing.assert_allclose(u, 0.4, rtol=1e-6)
        # v = hv/h = 1/5 = 0.2
        np.testing.assert_allclose(v, 0.2, rtol=1e-6)
        assert mask.all()  # all cells should be wet

    def test_compute_velocities_dry_cells(self):
        """Test velocity computation with dry cells."""
        h = np.ones((10, 10)) * 5.0
        h[0, 0] = 1e-8  # make one cell very shallow
        hu = np.ones((10, 10)) * 2.0
        hv = np.ones((10, 10)) * 1.0

        u, v, mask = WindForcing.compute_velocities(h, hu, hv, threshold=1e-6)

        # Dry cell should not be masked
        assert not mask[0, 0]
        # Wet cells should have computed velocities
        assert np.isclose(u[1, 1], 0.4, rtol=1e-6)

    def test_compute_velocities_zero_depth(self):
        """Test velocity computation with zero-depth cells."""
        h = np.zeros((10, 10))
        hu = np.zeros((10, 10))
        hv = np.zeros((10, 10))

        u, v, mask = WindForcing.compute_velocities(h, hu, hv)

        # All cells should be dry
        assert not mask.any()
        # Velocities should be zero
        np.testing.assert_allclose(u, 0.0, atol=1e-10)
        np.testing.assert_allclose(v, 0.0, atol=1e-10)

    def test_compute_velocities_array_broadcast(self):
        """Test velocity computation with broadcasting."""
        h = np.linspace(1.0, 10.0, 20).reshape(4, 5)
        hu = np.ones_like(h) * 2.0
        hv = np.ones_like(h) * 1.0

        u, v, mask = WindForcing.compute_velocities(h, hu, hv)

        assert u.shape == h.shape
        assert v.shape == h.shape
        # Check first element: u = 2/1 = 2
        assert np.isclose(u[0, 0], 2.0)

    def test_compute_wind_stress_magnitude(self):
        """Test wind stress computation."""
        wind_forcing = _make_wind_forcing(u_wind=5.0, v_wind=0.0)
        h = np.ones((10, 10)) * 5.0
        u = np.zeros((10, 10))
        v = np.zeros((10, 10))
        u_wind, v_wind = wind_forcing.get_wind()

        tau_x, tau_y = wind_forcing.compute_wind_stress(h, u, v, u_wind, v_wind)

        # Stress should be positive everywhere (in direction of wind)
        assert (tau_x >= 0).all()  # u_wind > 0
        assert (tau_y == 0).all()  # v_wind == 0

    def test_compute_wind_stress_zero_wind(self):
        """Test wind stress with zero wind."""
        wind = _make_wind_forcing(u_wind=0.0, v_wind=0.0)
        h = np.ones((10, 10)) * 5.0
        u = np.zeros((10, 10))
        v = np.zeros((10, 10))
        u_wind, v_wind = wind.get_wind()

        tau_x, tau_y = wind.compute_wind_stress(h, u, v, u_wind, v_wind)

        # No stress with no wind
        np.testing.assert_allclose(tau_x, 0.0, atol=1e-10)
        np.testing.assert_allclose(tau_y, 0.0, atol=1e-10)

    def test_compute_wind_stress_opposing_currents(self):
        """Test wind stress with opposing water currents."""
        wind_forcing = _make_wind_forcing(u_wind=5.0, v_wind=0.0)
        h = np.ones((10, 10)) * 5.0
        u = np.ones((10, 10)) * (-10.0)  # opposite to wind
        v = np.zeros((10, 10))
        u_wind, v_wind = wind_forcing.get_wind()

        tau_x, tau_y = wind_forcing.compute_wind_stress(h, u, v, u_wind, v_wind)

        # Relative wind is stronger when water moves opposite
        assert (tau_x >= 0).all()

    @pytest.mark.parametrize(
        "u_wind,v_wind",
        [
            (5.0, 0.0),
            (0.0, 5.0),
            (5.0, 5.0),
            (3.0, 4.0),
            (-5.0, 5.0),
        ],
    )
    def test_wind_directions_parametrized(self, u_wind, v_wind):
        """Test wind forcing with various wind directions."""
        wind = _make_wind_forcing(u_wind=u_wind, v_wind=v_wind)
        h = np.ones((10, 10)) * 5.0
        u = np.zeros((10, 10))
        v = np.zeros((10, 10))
        u_field, v_field = wind.get_wind()

        tau_x, tau_y = wind.compute_wind_stress(h, u, v, u_field, v_field)

        # Wind stress magnitude should match wind direction
        expected_u_sign = 1 if u_wind >= 0 else -1
        expected_v_sign = 1 if v_wind >= 0 else -1

        if u_wind != 0:
            assert (tau_x * expected_u_sign >= 0).all()
        if v_wind != 0:
            assert (tau_y * expected_v_sign >= 0).all()
