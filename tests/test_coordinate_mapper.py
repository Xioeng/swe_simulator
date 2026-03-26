"""Tests for coordinate mapper module."""

import numpy as np
import pytest

from tidalflow.coordinate_mapper import GeographicCoordinateMapper


def test_mapper_initialization(coordinate_mapper):
    """Test mapper initializes with correct center."""
    assert coordinate_mapper.lon0 == 0.0
    assert coordinate_mapper.lat0 == 0.0
    assert coordinate_mapper.R > 0


def test_mapper_custom_center():
    """Test mapper with custom center coordinates."""
    mapper = GeographicCoordinateMapper(lon0=-80.5, lat0=25.5)
    assert mapper.lon0 == -80.5
    assert mapper.lat0 == 25.5


def test_coord_to_metric_single_point(coordinate_mapper):
    """Test conversion of single point."""
    x_m, y_m = coordinate_mapper.coord_to_metric(0.0, 0.0)
    assert np.isclose(x_m, 0.0, atol=1e-6)
    assert np.isclose(y_m, 0.0, atol=1e-6)


def test_coord_to_metric_offset(coordinate_mapper):
    """Test conversion with offset from center."""
    x_m, y_m = coordinate_mapper.coord_to_metric(0.01, 0.01)
    assert x_m != 0.0
    assert y_m != 0.0


def test_coord_to_metric_array(coordinate_mapper):
    """Test conversion of coordinate arrays."""
    lons = np.array([0.0, 0.1, -0.1])
    lats = np.array([0.0, 0.1, -0.1])
    x_m, y_m = coordinate_mapper.coord_to_metric(lons, lats)

    assert x_m.shape == lons.shape
    assert y_m.shape == lats.shape
    assert np.isclose(x_m[0], 0.0, atol=1e-6)
    assert np.isclose(y_m[0], 0.0, atol=1e-6)


def test_coord_to_metric_meshgrid(coordinate_mapper):
    """Test conversion of meshgrid coordinates."""
    lon = np.linspace(-0.1, 0.1, 11)
    lat = np.linspace(-0.1, 0.1, 11)
    LON, LAT = np.meshgrid(lon, lat)

    X, Y = coordinate_mapper.coord_to_metric(LON, LAT)

    assert X.shape == LON.shape
    assert Y.shape == LAT.shape
    assert np.isclose(X[5, 5], 0.0, atol=1)
    assert np.isclose(Y[5, 5], 0.0, atol=1)


def test_metric_to_coord_single_point(coordinate_mapper):
    """Test conversion from metric back to geographic."""
    lon_orig, lat_orig = 0.0, 0.0
    x_m, y_m = coordinate_mapper.coord_to_metric(lon_orig, lat_orig)
    lon_back, lat_back = coordinate_mapper.metric_to_coord(x_m, y_m)

    assert np.isclose(lon_back, lon_orig, atol=1e-6)
    assert np.isclose(lat_back, lat_orig, atol=1e-6)


def test_round_trip_conversion(coordinate_mapper):
    """Test round-trip conversion (geo → metric → geo)."""
    lons = np.array([-0.05, 0.0, 0.05])
    lats = np.array([-0.05, 0.0, 0.05])

    x_m, y_m = coordinate_mapper.coord_to_metric(lons, lats)
    lon_back, lat_back = coordinate_mapper.metric_to_coord(x_m, y_m)

    np.testing.assert_allclose(lon_back, lons, rtol=1e-5)
    np.testing.assert_allclose(lat_back, lats, rtol=1e-5)


def test_coordinate_distance(coordinate_mapper):
    """Test distance calculations."""
    x1, y1 = coordinate_mapper.coord_to_metric(0.0, 0.0)
    x2, y2 = coordinate_mapper.coord_to_metric(1.0, 0.0)

    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    expected_order = coordinate_mapper.R * np.radians(1.0)
    assert 0.5 * expected_order < distance < 1.5 * expected_order


def test_coordinate_symmetry(coordinate_mapper):
    """Test that transformations are symmetric around origin."""
    x_pos, y_pos = coordinate_mapper.coord_to_metric(0.1, 0.0)
    x_neg, y_neg = coordinate_mapper.coord_to_metric(-0.1, 0.0)

    assert np.isclose(x_pos, -x_neg, rtol=1e-5)
    assert np.isclose(y_pos, y_neg, rtol=1e-5)


@pytest.mark.parametrize(
    "lon,lat",
    [
        (0.0, 0.0),
        (0.05, 0.05),
        (-0.05, -0.05),
        (0.1, -0.05),
        (-0.1, 0.05),
    ],
)
def test_round_trip_parametrized(coordinate_mapper, lon, lat):
    """Parametrized test for multiple coordinate pairs."""
    x_m, y_m = coordinate_mapper.coord_to_metric(lon, lat)
    lon_back, lat_back = coordinate_mapper.metric_to_coord(x_m, y_m)

    assert np.isclose(lon_back, lon, atol=1e-6)
    assert np.isclose(lat_back, lat, atol=1e-6)
