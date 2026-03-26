"""Tests for swe_simulator.utils.bathymetry."""

import sys
import types

import numpy as np
import numpy.typing as npt
import pytest

from swe_simulator.utils import bathymetry


def test_load_gebco_data_from_xarray_transposes_elevation(tmp_path, monkeypatch):
    """GEBCO loader should transpose elevation to (lon, lat) shape."""
    nc_path = tmp_path / "fake_gebco.nc"
    nc_path.write_text("placeholder", encoding="utf-8")

    lon = np.array([100.0, 101.0, 102.0])
    lat = np.array([20.0, 21.0])
    elevation_lat_lon = np.array(
        [
            [-5.0, -6.0, -7.0],
            [-8.0, -9.0, -10.0],
        ]
    )

    fake_dataset = types.SimpleNamespace(
        variables={"lon": None, "lat": None, "elevation": None},
        lon=types.SimpleNamespace(values=lon),
        lat=types.SimpleNamespace(values=lat),
        elevation=types.SimpleNamespace(values=elevation_lat_lon),
    )

    fake_xarray = types.SimpleNamespace(open_dataset=lambda _: fake_dataset)
    monkeypatch.setitem(sys.modules, "xarray", fake_xarray)

    data = bathymetry.load_gebco_data(nc_path)

    np.testing.assert_allclose(data["lon"], lon)
    np.testing.assert_allclose(data["lat"], lat)
    assert data["elevation"].shape == (len(lon), len(lat))
    np.testing.assert_allclose(data["elevation"], elevation_lat_lon.T)


def test_build_gebco_interpolator_interpolates_lon_lat_nodes(monkeypatch):
    """GEBCO interpolator should query in (lon, lat) order."""
    lon = np.array([0.0, 1.0])
    lat = np.array([10.0, 20.0])
    elevation_lon_lat = np.array(
        [
            [100.0, 200.0],
            [110.0, 210.0],
        ]
    )

    monkeypatch.setattr(
        bathymetry,
        "load_gebco_data",
        lambda _: {
            "lon": lon,
            "lat": lat,
            "elevation": elevation_lon_lat,
        },
    )

    interpolator = bathymetry.build_gebco_interpolator("ignored.nc")
    points = np.array(
        [
            [0.0, 10.0],
            [0.0, 20.0],
            [1.0, 10.0],
            [1.0, 20.0],
        ]
    )
    expected = np.array([100.0, 200.0, 110.0, 210.0])

    result = interpolator(points)
    np.testing.assert_allclose(result, expected)


def test_interpolate_gebco_on_grid_delegates_to_grid_interpolator(
    monkeypatch,
):
    """GEBCO grid interpolation should use the built interpolator on mesh."""

    def fake_interpolator(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # f(lon, lat) = lon - lat
        return points[:, 0] - points[:, 1]

    monkeypatch.setattr(
        bathymetry,
        "build_gebco_interpolator",
        lambda *args, **kwargs: fake_interpolator,
    )

    X, Y = np.meshgrid(
        np.array([0.0, 2.0]),
        np.array([10.0, 20.0]),
        indexing="xy",
    )

    result = bathymetry.interpolate_gebco_on_grid(
        X=X,
        Y=Y,
        nc_path="ignored.nc",
    )
    expected = X - Y

    np.testing.assert_allclose(result, expected)


def test_interpolate_gebco_on_grid_shape_mismatch_raises():
    """GEBCO grid interpolation should validate X/Y shape compatibility."""
    X = np.zeros((3, 2))
    Y = np.zeros((2, 3))

    with pytest.raises(ValueError, match="same shape"):
        bathymetry.interpolate_gebco_on_grid(
            X=X,
            Y=Y,
            nc_path="ignored.nc",
        )
