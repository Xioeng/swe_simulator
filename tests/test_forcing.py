"""Tests for wind forcing module."""

import numpy as np
import pytest

from tidalflow.forcing import WindForcing


def test_wind_forcing_initialization(wind_forcing):
    """Test basic initialization."""
    u, v = wind_forcing.get_wind()
    assert np.allclose(u, 5.0)
    assert np.allclose(v, 2.0)


@pytest.mark.parametrize(
    "wind_forcing",
    [{"u_wind": 0.0, "v_wind": 0.0}],
    indirect=True,
)
def test_wind_forcing_defaults(wind_forcing):
    """Test default values."""
    u, v = wind_forcing.get_wind()
    assert np.allclose(u, 0.0)
    assert np.allclose(v, 0.0)
    assert wind_forcing.c_d > 0
    assert wind_forcing.rho_air > 0
    assert wind_forcing.rho_water > 0


def test_wind_forcing_custom_density(wind_forcing_factory):
    """Test custom density values."""
    wind = wind_forcing_factory(rho_air=1.3, rho_water=1025.0)
    assert wind.rho_air == 1.3
    assert wind.rho_water == 1025.0


def test_get_wind_velocity(wind_forcing_factory):
    """Test getting wind velocity."""
    wind_forcing = wind_forcing_factory(u_wind=5.0, v_wind=0.0)
    u, v = wind_forcing.get_wind()
    assert u.shape == v.shape
    assert np.allclose(u, 5.0)
    assert np.allclose(v, 0.0)


def test_set_drag_coefficient(wind_forcing):
    """Test setting drag coefficient."""
    wind_forcing.set_drag_coefficient(1.5e-3)
    assert wind_forcing.get_drag_coefficient() == 1.5e-3


def test_compute_velocities_flat_water():
    """Test velocity computation for uniform water."""
    h = np.ones((10, 10)) * 5.0
    hu = np.ones((10, 10)) * 2.0
    hv = np.ones((10, 10)) * 1.0

    u, v, mask = WindForcing.compute_velocities(h, hu, hv)

    np.testing.assert_allclose(u, 0.4, rtol=1e-6)
    np.testing.assert_allclose(v, 0.2, rtol=1e-6)
    assert mask.all()


def test_compute_velocities_dry_cells():
    """Test velocity computation with dry cells."""
    h = np.ones((10, 10)) * 5.0
    h[0, 0] = 1e-8
    hu = np.ones((10, 10)) * 2.0
    hv = np.ones((10, 10)) * 1.0

    u, v, mask = WindForcing.compute_velocities(h, hu, hv, threshold=1e-6)

    assert not mask[0, 0]
    assert np.isclose(u[1, 1], 0.4, rtol=1e-6)


def test_compute_velocities_zero_depth():
    """Test velocity computation with zero-depth cells."""
    h = np.zeros((10, 10))
    hu = np.zeros((10, 10))
    hv = np.zeros((10, 10))

    u, v, mask = WindForcing.compute_velocities(h, hu, hv)

    assert not mask.any()
    np.testing.assert_allclose(u, 0.0, atol=1e-10)
    np.testing.assert_allclose(v, 0.0, atol=1e-10)


def test_compute_velocities_array_broadcast():
    """Test velocity computation with broadcasting."""
    h = np.linspace(1.0, 10.0, 20).reshape(4, 5)
    hu = np.ones_like(h) * 2.0
    hv = np.ones_like(h) * 1.0

    u, v, mask = WindForcing.compute_velocities(h, hu, hv)

    assert u.shape == h.shape
    assert v.shape == h.shape
    assert np.isclose(u[0, 0], 2.0)


def test_compute_wind_stress_magnitude(wind_forcing_factory):
    """Test wind stress computation."""
    wind_forcing = wind_forcing_factory(u_wind=5.0, v_wind=0.0)
    h = np.ones((10, 10)) * 5.0
    u = np.zeros((10, 10))
    v = np.zeros((10, 10))
    u_wind, v_wind = wind_forcing.get_wind()

    tau_x, tau_y = wind_forcing.compute_wind_stress(h, u, v, u_wind, v_wind)

    assert (tau_x >= 0).all()
    assert (tau_y == 0).all()


def test_compute_wind_stress_zero_wind(wind_forcing_factory):
    """Test wind stress with zero wind."""
    wind = wind_forcing_factory(u_wind=0.0, v_wind=0.0)
    h = np.ones((10, 10)) * 5.0
    u = np.zeros((10, 10))
    v = np.zeros((10, 10))
    u_wind, v_wind = wind.get_wind()

    tau_x, tau_y = wind.compute_wind_stress(h, u, v, u_wind, v_wind)

    np.testing.assert_allclose(tau_x, 0.0, atol=1e-10)
    np.testing.assert_allclose(tau_y, 0.0, atol=1e-10)


def test_compute_wind_stress_opposing_currents(wind_forcing_factory):
    """Test wind stress with opposing water currents."""
    wind_forcing = wind_forcing_factory(u_wind=5.0, v_wind=0.0)
    h = np.ones((10, 10)) * 5.0
    u = np.ones((10, 10)) * (-10.0)
    v = np.zeros((10, 10))
    u_wind, v_wind = wind_forcing.get_wind()

    tau_x, tau_y = wind_forcing.compute_wind_stress(h, u, v, u_wind, v_wind)

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
def test_wind_directions_parametrized(u_wind, v_wind, wind_forcing_factory):
    """Test wind forcing with various wind directions."""
    wind = wind_forcing_factory(u_wind=u_wind, v_wind=v_wind)
    h = np.ones((10, 10)) * 5.0
    u = np.zeros((10, 10))
    v = np.zeros((10, 10))
    u_field, v_field = wind.get_wind()

    tau_x, tau_y = wind.compute_wind_stress(h, u, v, u_field, v_field)

    expected_u_sign = 1 if u_wind >= 0 else -1
    expected_v_sign = 1 if v_wind >= 0 else -1

    if u_wind != 0:
        assert (tau_x * expected_u_sign >= 0).all()
    if v_wind != 0:
        assert (tau_y * expected_v_sign >= 0).all()
