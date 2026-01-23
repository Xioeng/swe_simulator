"""
Validation utilities for input data.

This module provides functions for validating arrays and checking
for common data issues like NaN values, infinite values, and
incorrect shapes.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def validate_array_shape(
    array: npt.NDArray,
    expected_shape: Tuple[int, ...],
    name: str = "array",
    allow_none: bool = False,
) -> None:
    """
    Validate that array has expected shape.

    Parameters
    ----------
    array : np.ndarray or None
        Array to validate
    expected_shape : tuple of int
        Expected shape (e.g., (3, 100, 100))
    name : str, default="array"
        Name of array for error messages
    allow_none : bool, default=False
        Whether to allow None values

    Raises
    ------
    ValueError
        If array shape doesn't match expected shape
    TypeError
        If array is None and allow_none is False

    Examples
    --------
    >>> arr = np.zeros((3, 100, 100))
    >>> validate_array_shape(arr, (3, 100, 100), name="initial_condition")
    >>> # No error - shape matches
    >>>
    >>> validate_array_shape(arr, (2, 100, 100), name="initial_condition")
    ValueError: initial_condition has shape (3, 100, 100), expected (2, 100, 100)
    """
    if array is None:
        if allow_none:
            return
        else:
            raise TypeError(
                f"{name} is None, expected array with shape {expected_shape}"
            )

    if array.shape != expected_shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {expected_shape}")

    logger.debug(f"{name} shape validation passed: {array.shape}")


def check_for_nans(
    array: npt.NDArray,
    name: str = "array",
    allow_nans: bool = False,
) -> int:
    """
    Check if array contains NaN values.

    Parameters
    ----------
    array : np.ndarray
        Array to check
    name : str, default="array"
        Name of array for error/log messages
    allow_nans : bool, default=False
        Whether to allow NaN values

    Returns
    -------
    int
        Number of NaN values found

    Raises
    ------
    ValueError
        If array contains NaN values and allow_nans is False

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, np.nan, 4.0])
    >>> check_for_nans(arr, name="data", allow_nans=True)
    1
    >>>
    >>> check_for_nans(arr, name="data", allow_nans=False)
    ValueError: data contains 1 NaN values at 25.00% of elements
    """
    n_nans = np.sum(np.isnan(array))

    if n_nans > 0:
        pct_nans = 100 * n_nans / array.size
        message = f"{name} contains {n_nans} NaN values at {pct_nans:.2f}% of elements"

        if allow_nans:
            logger.warning(message)
        else:
            raise ValueError(message)

    else:
        logger.debug(f"{name} contains no NaN values")

    return n_nans


def check_for_infs(
    array: npt.NDArray,
    name: str = "array",
) -> int:
    """
    Check if array contains infinite values.

    Parameters
    ----------
    array : np.ndarray
        Array to check
    name : str, default="array"
        Name of array for error messages

    Returns
    -------
    int
        Number of infinite values found

    Raises
    ------
    ValueError
        If array contains infinite values

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, np.inf, 4.0])
    >>> check_for_infs(arr, name="velocities")
    ValueError: velocities contains 1 infinite values at 25.00% of elements
    """
    n_infs = np.sum(np.isinf(array))

    if n_infs > 0:
        pct_infs = 100 * n_infs / array.size
        message = (
            f"{name} contains {n_infs} infinite values at {pct_infs:.2f}% of elements"
        )
        raise ValueError(message)

    logger.debug(f"{name} contains no infinite values")
    return n_infs


def validate_range(
    array: npt.NDArray,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    name: str = "array",
) -> None:
    """
    Validate that array values are within specified range.

    Parameters
    ----------
    array : np.ndarray
        Array to validate
    min_value : float, optional
        Minimum allowed value (inclusive)
    max_value : float, optional
        Maximum allowed value (inclusive)
    name : str, default="array"
        Name of array for error messages

    Raises
    ------
    ValueError
        If any values are outside the specified range

    Examples
    --------
    >>> depths = np.array([0.1, 1.0, 5.0, 10.0])
    >>> validate_range(depths, min_value=0.0, max_value=100.0, name="water_depth")
    >>> # No error
    >>>
    >>> validate_range(depths, min_value=1.0, name="water_depth")
    ValueError: water_depth contains values below minimum 1.0: min=0.1
    """
    if min_value is not None:
        actual_min = np.min(array)
        if actual_min < min_value:
            raise ValueError(
                f"{name} contains values below minimum {min_value}: min={actual_min}"
            )

    if max_value is not None:
        actual_max = np.max(array)
        if actual_max > max_value:
            raise ValueError(
                f"{name} contains values above maximum {max_value}: max={actual_max}"
            )

    logger.debug(
        f"{name} range validation passed: [{np.min(array):.2e}, {np.max(array):.2e}]"
    )


def validate_positive(
    array: npt.NDArray,
    name: str = "array",
    strict: bool = False,
) -> None:
    """
    Validate that array contains only positive values.

    Parameters
    ----------
    array : np.ndarray
        Array to validate
    name : str, default="array"
        Name of array for error messages
    strict : bool, default=False
        If True, values must be > 0. If False, values must be >= 0.

    Raises
    ------
    ValueError
        If array contains non-positive values

    Examples
    --------
    >>> depths = np.array([0.0, 1.0, 5.0, 10.0])
    >>> validate_positive(depths, name="water_depth", strict=False)
    >>> # No error (0 is allowed)
    >>>
    >>> validate_positive(depths, name="water_depth", strict=True)
    ValueError: water_depth contains non-positive values: min=0.0
    """
    min_val = np.min(array)

    if strict:
        if min_val <= 0:
            raise ValueError(
                f"{name} contains non-positive values: min={min_val}. "
                "All values must be > 0."
            )
    else:
        if min_val < 0:
            raise ValueError(
                f"{name} contains negative values: min={min_val}. "
                "All values must be >= 0."
            )

    logger.debug(f"{name} positivity validation passed: min={min_val}")


def validate_bathymetry(
    bathymetry: npt.NDArray[np.float64],
    grid_shape: Tuple[int, int],
    allow_land: bool = True,
) -> None:
    """
    Validate bathymetry array for use in shallow water simulations.

    This is a convenience function that performs multiple common
    validation checks on bathymetry data.

    Parameters
    ----------
    bathymetry : np.ndarray
        Bathymetry array (negative values indicate depth)
    grid_shape : tuple of int
        Expected grid shape (ny, nx)
    allow_land : bool, default=True
        Whether to allow positive values (land above sea level)

    Raises
    ------
    ValueError
        If bathymetry fails validation checks

    Examples
    --------
    >>> bathymetry = -10.0 * np.ones((100, 100))
    >>> validate_bathymetry(bathymetry, grid_shape=(100, 100))
    >>> # Passes all checks
    """
    # Check shape
    validate_array_shape(bathymetry, grid_shape, name="bathymetry")

    # Check for invalid values
    check_for_nans(bathymetry, name="bathymetry", allow_nans=False)
    check_for_infs(bathymetry, name="bathymetry")

    # Check for unrealistic values
    min_depth = np.min(bathymetry)
    max_elevation = np.max(bathymetry)

    # Warn about very deep water (>11000m, deeper than Mariana Trench)
    if min_depth < -11000:
        logger.warning(
            f"Bathymetry contains very deep values: {min_depth:.2f}m. "
            "This is deeper than the Mariana Trench. Check your data."
        )

    # Warn/error about land
    if max_elevation > 0:
        if allow_land:
            logger.info(
                f"Bathymetry contains land (positive values) up to {max_elevation:.2f}m. "
                "These areas will be treated as obstacles."
            )
        else:
            raise ValueError(
                f"Bathymetry contains land (positive values) up to {max_elevation:.2f}m. "
                "Set allow_land=True if this is intentional."
            )

    logger.info(
        f"Bathymetry validation passed: "
        f"depth range [{min_depth:.2f}, {max_elevation:.2f}]m"
    )
