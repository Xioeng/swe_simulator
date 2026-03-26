"""Tests for tidalflow.utils.grid."""

import numpy as np
import numpy.typing as npt
import pytest

from tidalflow.utils import grid


def test_generate_cell_centers_shape_and_midpoints():
    """Cell centers should have expected shape and midpoint coordinates."""
    X, Y = grid.generate_cell_centers(
        x_lower=0.0,
        x_upper=2.0,
        y_lower=10.0,
        y_upper=14.0,
        nx=2,
        ny=2,
    )

    assert X.shape == (2, 2)
    assert Y.shape == (2, 2)
    np.testing.assert_allclose(X[:, 0], np.array([0.5, 1.5]))
    np.testing.assert_allclose(Y[0, :], np.array([11.0, 13.0]))


def test_build_regular_grid_interpolator_exact_nodes_lon_lat_order():
    """Regular interpolator should honor (lon, lat) query ordering."""
    lon = np.array([0.0, 1.0])
    lat = np.array([10.0, 20.0])

    # values indexed as values[lon_index, lat_index]
    values = np.array(
        [
            [10.0, 20.0],
            [11.0, 21.0],
        ]
    )

    interpolator = grid.build_regular_grid_interpolator(
        lon=lon,
        lat=lat,
        values=values,
        method="linear",
    )

    points = np.array(
        [
            [0.0, 10.0],
            [0.0, 20.0],
            [1.0, 10.0],
            [1.0, 20.0],
        ]
    )
    expected = np.array([10.0, 20.0, 11.0, 21.0])
    result = interpolator(points)

    np.testing.assert_allclose(result, expected)


def test_build_regular_grid_interpolator_shape_validation():
    """Regular interpolator should reject incompatible value shapes."""
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 1.0])
    bad_values = np.zeros((2, 3))

    with pytest.raises(ValueError, match="values shape"):
        grid.build_regular_grid_interpolator(lon, lat, bad_values)


def test_build_scattered_interpolator_reproduces_sample_points():
    """Scattered interpolator should reproduce known sample values."""
    lon = np.array([0.0, 1.0, 0.0, 1.0])
    lat = np.array([0.0, 0.0, 1.0, 1.0])
    values = lon + 2.0 * lat

    interpolator = grid.build_scattered_interpolator(
        lon=lon,
        lat=lat,
        values=values,
        method="linear",
        use_nearest_fallback=True,
    )

    points = np.column_stack([lon, lat])
    result = interpolator(points)
    np.testing.assert_allclose(result, values)


def test_interpolate_on_mesh_uses_lon_lat_query_order():
    """Mesh interpolation should pass points to interpolator as (lon, lat)."""

    def fake_interpolator(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # f(lon, lat) = lon + 2*lat
        return points[:, 0] + 2.0 * points[:, 1]

    X, Y = np.meshgrid(
        np.array([0.0, 1.0]),
        np.array([10.0, 20.0]),
        indexing="ij",
    )

    result = grid.interpolate_on_mesh(fake_interpolator, X, Y)
    expected = X + 2.0 * Y

    np.testing.assert_allclose(result, expected)
