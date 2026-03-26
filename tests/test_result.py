"""Tests for result module."""

import numpy as np

from tidalflow.result import SWEResult


def test_to_dict_returns_expected_structure(basic_config):
    """Test dataclass conversion to dictionary."""
    result = SWEResult(
        meshgrid_coord=(np.array([[0.0]]), np.array([[1.0]])),
        meshgrid_metric=(np.array([[10.0]]), np.array([[20.0]])),
        solution=np.zeros((1, 3, 1, 1)),
        bathymetry=np.array([[-5.0]]),
        initial_condition=np.array([[[1.0]], [[0.0]], [[0.0]]]),
        wind_forcing=(np.array([[5.0]]), np.array([[2.0]])),
        config=basic_config,
    )

    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert "solution" in result_dict
    assert "bathymetry" in result_dict
    assert "wind_forcing" in result_dict
    assert isinstance(result_dict["config"], dict)
    assert result_dict["config"]["nx"] == basic_config.nx


def test_save_and_load_round_trip(tmp_path, basic_config):
    """Test result persistence with pickle save/load."""
    mesh_x = np.linspace(0.0, 1.0, 4).reshape(2, 2)
    mesh_y = np.linspace(1.0, 2.0, 4).reshape(2, 2)
    solution = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
    bathymetry = -10.0 * np.ones((2, 2), dtype=float)
    initial_condition = np.zeros((3, 2, 2), dtype=float)
    wind_u = np.full((2, 2), 5.0)
    wind_v = np.full((2, 2), 2.0)

    original = SWEResult(
        meshgrid_coord=(mesh_x, mesh_y),
        meshgrid_metric=(mesh_x * 1000.0, mesh_y * 1000.0),
        solution=solution,
        bathymetry=bathymetry,
        initial_condition=initial_condition,
        wind_forcing=(wind_u, wind_v),
        config=basic_config,
    )

    out_file = tmp_path / "result.pkl"
    original.save(out_file)
    loaded = SWEResult.load(out_file)

    assert isinstance(loaded, SWEResult)
    assert loaded.config == basic_config
    np.testing.assert_allclose(loaded.solution, solution)
    np.testing.assert_allclose(loaded.bathymetry, bathymetry)
    np.testing.assert_allclose(loaded.initial_condition, initial_condition)

    assert loaded.meshgrid_coord is not None
    loaded_x, loaded_y = loaded.meshgrid_coord
    np.testing.assert_allclose(loaded_x, mesh_x)
    np.testing.assert_allclose(loaded_y, mesh_y)

    assert loaded.wind_forcing is not None
    loaded_wind_u, loaded_wind_v = loaded.wind_forcing
    np.testing.assert_allclose(loaded_wind_u, wind_u)
    np.testing.assert_allclose(loaded_wind_v, wind_v)


def test_save_accepts_string_path(tmp_path):
    """Test save/load work with string filepaths."""
    result = SWEResult(solution=np.zeros((1, 3, 1, 1)))
    out_file = tmp_path / "result_string.pkl"

    result.save(str(out_file))
    loaded = SWEResult.load(str(out_file))

    assert loaded.solution is not None
    np.testing.assert_allclose(loaded.solution, result.solution)
