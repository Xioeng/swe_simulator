"""
Input/Output utilities for reading and writing simulation data.

This module provides functions for:
- Reading PyClaw solution files
- Loading bathymetry and coordinate grids
- Saving simulation results
- Managing output directories
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from ..logging_config import get_logger
from ..result import SWEResult

logger = get_logger(__name__)


def get_frame_count(output_path: Union[str, Path]) -> int:
    """
    Count the number of solution frames in an output directory.

    Parameters
    ----------
    output_path : str or Path
        Path to output directory containing PyClaw solution files

    Returns
    -------
    int
        Number of solution frames found

    Notes
    -----
    Looks for files matching the pattern "claw.ptc*" (excluding .info files)
    which are PETSc binary format solution files from PyClaw.
    """
    output_path = Path(output_path)

    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_path}")
        return 0

    try:
        files = os.listdir(output_path)
        frame_files = [
            f for f in files if f.startswith("claw.ptc") and not f.endswith(".info")
        ]
        n_frames = len(frame_files)

        logger.debug(f"Found {n_frames} solution frames in {output_path}")
        return n_frames

    except Exception as e:
        logger.error(f"Error counting frames in {output_path}: {e}")
        return 0


def load_bathymetry_and_meshgrid(
    output_path: Union[str, Path],
) -> Tuple[
    npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
]:
    """
    Load bathymetry and coordinate meshgrid from output directory.

    Parameters
    ----------
    output_path : str or Path
        Path to output directory

    Returns
    -------
    bathymetry : np.ndarray
        2D array of bathymetry values (meters, negative = depth)
    meshgrid : tuple of np.ndarray
        Tuple (X, Y) where X and Y are 2D coordinate arrays:
        - X: longitude coordinates (degrees)
        - Y: latitude coordinates (degrees)

    Raises
    ------
    FileNotFoundError
        If required files (bathymetry.npy, coord_meshgrid.npy) are not found
    ValueError
        If loaded data has unexpected format

    Notes
    -----
    This function expects the output directory to contain:
    - bathymetry.npy: 2D array of bathymetry values
    - coord_meshgrid.npy: Array with shape (2, ny, nx) containing [X, Y]
    """
    output_path = Path(output_path)

    # Load coordinate meshgrid
    coord_file = output_path / "coord_meshgrid.npy"
    if not coord_file.exists():
        raise FileNotFoundError(
            f"Coordinate meshgrid file not found: {coord_file}\n"
            "Make sure the simulation has been run and output was saved."
        )

    try:
        coord_data = np.load(coord_file, allow_pickle=True)
        if coord_data.shape[0] != 2:
            raise ValueError(
                f"Expected coord_meshgrid.npy to have shape (2, ny, nx), "
                f"got {coord_data.shape}"
            )
        X, Y = coord_data[0], coord_data[1]
        logger.debug(f"Loaded coordinate grid with shape {X.shape}")
    except Exception as e:
        raise ValueError(f"Error loading coordinate meshgrid: {e}")

    # Load bathymetry
    bathy_file = output_path / "bathymetry.npy"
    if not bathy_file.exists():
        raise FileNotFoundError(
            f"Bathymetry file not found: {bathy_file}\n"
            "Make sure the simulation has been run and output was saved."
        )

    try:
        bathymetry = np.load(bathy_file)
        if bathymetry.shape != X.shape:
            raise ValueError(
                f"Bathymetry shape {bathymetry.shape} does not match "
                f"coordinate grid shape {X.shape}"
            )
        logger.debug(
            f"Loaded bathymetry: min={bathymetry.min():.2f}m, "
            f"max={bathymetry.max():.2f}m"
        )
    except Exception as e:
        raise ValueError(f"Error loading bathymetry: {e}")

    logger.info(f"Loaded bathymetry and meshgrid from {output_path}")
    return bathymetry, (X, Y)


def read_solutions(
    outdir: Union[str, Path] = "_output",
    frames_list: Optional[List[int]] = None,
    read_aux: bool = False,
) -> Dict[str, Any]:
    """
    Read PyClaw solution files from output directory.

    This function loads simulation results including solution states,
    bathymetry, and coordinate grids from a PyClaw output directory.

    Parameters
    ----------
    outdir : str or Path, default="_output"
        Path to output directory containing solution files
    frames_list : List[int], optional
        List of frame numbers to read. If None, reads all available frames.
    read_aux : bool, default=False
        Whether to read auxiliary variables (e.g., bathymetry from each frame)

    Returns
    -------
    dict
        Dictionary containing:
        - 'solutions': Array of shape (n_frames, n_vars, ny, nx)
          where n_vars is typically 3 for [h, hu, hv]
        - 'bathymetry': 2D array of bathymetry values
        - 'meshgrid': Tuple (X, Y) of coordinate arrays
        - 'times': Array of solution times (if available)
        - 'frames': List of frame numbers that were read

    Raises
    ------
    FileNotFoundError
        If output directory doesn't exist
    ImportError
        If clawpack.petclaw is not available

    Notes
    -----
    - Solutions are stored in PETSc binary format (.ptc files)
    - The function automatically detects available frames if frames_list is None
    - Missing frames are skipped with a warning
    - Returns empty arrays if no valid solutions found
    """
    try:
        import clawpack.petclaw as pyclaw
    except ImportError:
        raise ImportError(
            "clawpack.petclaw is required to read solutions.\n"
            "Install with: pip install clawpack"
        )

    outdir = Path(outdir)

    if not outdir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {outdir}")

    logger.info(f"Reading solutions from {outdir}")

    # Load bathymetry and meshgrid
    result = SWEResult().load(outdir.joinpath("result.pkl"))
    bathymetry, meshgrid = result.bathymetry, result.meshgrid_coord

    # Determine which frames to read
    if frames_list is None:
        n_frames = get_frame_count(outdir)
        if n_frames == 0:
            logger.warning("No solution frames found")
            raise FileNotFoundError("No solution frames found in output directory")
        frames_list = list(range(n_frames))
        logger.info(f"Found {n_frames} frames to read")
    else:
        logger.info(f"Reading {len(frames_list)} specified frames")

    # Read solutions
    solutions = None
    times = None
    valid_frames = []

    for i, frame_num in enumerate(frames_list):
        try:
            # Read solution
            sol = pyclaw.Solution()
            sol.read(
                frame_num,
                path=str(outdir),
                file_prefix="claw",
                file_format="petsc",
                read_aux=read_aux,
            )

            # Initialize solutions array on first successful read
            if solutions is None:
                n_total = len(frames_list)
                solutions = np.empty((n_total, *sol.state.q.shape))
            if times is None:
                times = np.empty(n_total)

            # Store solution data
            solutions[i] = sol.state.q
            times[i] = sol.t
            valid_frames.append(frame_num)

            if (i + 1) % 10 == 0:
                logger.debug(f"Read {i + 1}/{len(frames_list)} frames")

        except Exception as e:
            logger.warning(f"Could not read frame {frame_num}: {e}")
            continue

    # Trim solutions array if some frames failed
    if solutions is not None and len(valid_frames) < len(frames_list):
        solutions = solutions[: len(valid_frames)]

    # Convert to arrays
    if solutions is not None:
        times = np.array(times)
        logger.info(
            f"Successfully read {len(valid_frames)} frames "
            f"(t={times[0]:.2f} to {times[-1]:.2f}s)"
        )
    else:
        solutions = np.array([])
        times = np.array([])
        logger.warning("No solutions were successfully read")

    return {
        "solutions": solutions,
        "bathymetry": bathymetry,
        "meshgrid": meshgrid,
        "times": times,
        "frames": valid_frames,
    }


def save_solution(
    solution: npt.NDArray[np.float64],
    filename: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    compress: bool = True,
) -> None:
    """
    Save solution array to file with optional metadata.

    Parameters
    ----------
    solution : np.ndarray
        Solution array to save
    filename : str or Path
        Output filename (will be saved as .npz)
    metadata : dict, optional
        Dictionary of metadata to save with solution
        (e.g., {'time': 100.0, 'dt': 1.0})
    compress : bool, default=True
        Whether to use compression (saves disk space)

    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        metadata = {}

    try:
        if compress:
            np.savez_compressed(filename, solution=solution, **metadata)
        else:
            np.savez(filename, solution=solution, **metadata)

        file_size = filename.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Solution saved to {filename} ({file_size:.2f} MB)")

    except Exception as e:
        logger.error(f"Error saving solution to {filename}: {e}")
        raise


def load_solution(filename: Union[str, Path]) -> Dict[str, Any]:
    """
    Load solution and metadata from file.

    Parameters
    ----------
    filename : str or Path
        Path to .npz file

    Returns
    -------
    dict
        Dictionary containing 'solution' and any saved metadata
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Solution file not found: {filename}")

    try:
        data = np.load(filename, allow_pickle=True)
        result = dict(data)
        logger.info(f"Loaded solution from {filename}")
        return result

    except Exception as e:
        logger.error(f"Error loading solution from {filename}: {e}")
        raise
