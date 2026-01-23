"""Configuration dataclass for SWE Simulator."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import clawpack.petclaw as pyclaw

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """
    Configuration container for shallow water equation simulations.

    Parameters
    ----------
    lon_range : Tuple[float, float], optional
        Longitude range in degrees (lon_min, lon_max)
    lat_range : Tuple[float, float], optional
        Latitude range in degrees (lat_min, lat_max)
    nx : int, optional
        Number of grid cells in x-direction
    ny : int, optional
        Number of grid cells in y-direction
    t_final : float, optional
        Final simulation time in seconds
    dt : float, optional
        Time step size in seconds
    gravity : float, default=9.81
        Gravitational acceleration in m/s²
    output_dir : str, default="_output"
        Directory for simulation output
    frame_interval : int, default=1
        Number of time steps between output frames
    cfl_desired : float, default=0.9
        Desired CFL number for adaptive time stepping
    cfl_max : float, default=1.0
        Maximum allowed CFL number
    num_output_times : int, optional
        Number of output times (overrides frame_interval if set)
    bc_lower : List[int], default=(0, 0)
        Boundary conditions for lower boundaries (x, y)
    bc_upper : List[int], default=(0, 0)
        Boundary conditions for upper boundaries (x, y)
    """

    lon_range: Optional[Tuple[float, float]] = None
    lat_range: Optional[Tuple[float, float]] = None
    nx: Optional[int] = None
    ny: Optional[int] = None
    t_final: Optional[float] = None
    dt: Optional[float] = None
    gravity: float = 9.81
    output_dir: str = "_output"
    frame_interval: int = 1
    cfl_desired: float = 0.9
    cfl_max: float = 1.0
    multiple_output_times: bool = False
    bc_lower: Tuple[int, int] = (pyclaw.BC.wall, pyclaw.BC.wall)
    bc_upper: Tuple[int, int] = (pyclaw.BC.wall, pyclaw.BC.wall)

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []

        # Domain
        if self.lon_range is None:
            errors.append("lon_range is required")
        elif len(self.lon_range) != 2 or self.lon_range[0] >= self.lon_range[1]:
            errors.append("lon_range must be (lon_min, lon_max) with lon_min < lon_max")

        if self.lat_range is None:
            errors.append("lat_range is required")
        elif len(self.lat_range) != 2 or self.lat_range[0] >= self.lat_range[1]:
            errors.append("lat_range must be (lat_min, lat_max) with lat_min < lat_max")

        if self.nx is None or self.nx <= 0:
            errors.append("nx must be a positive integer")
        if self.ny is None or self.ny <= 0:
            errors.append("ny must be a positive integer")

        # Time
        if self.t_final is not None and self.t_final <= 0:
            errors.append("t_final must be positive")
        if self.dt is not None and self.dt <= 0:
            errors.append("dt must be positive")

        # Physics
        if self.gravity <= 0:
            errors.append("gravity must be positive")

        # CFL
        if not 0 < self.cfl_desired <= 1.0:
            errors.append("cfl_desired must be in (0, 1]")
        if not 0 < self.cfl_max <= 1.0:
            errors.append("cfl_max must be in (0, 1]")
        if self.cfl_desired > self.cfl_max:
            errors.append("cfl_desired must be <= cfl_max")

        # Output
        if self.frame_interval <= 0:
            errors.append("frame_interval must be positive")
        if type(self.multiple_output_times) is not bool:
            errors.append("multiple_output_times must be a boolean")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

        logger.debug("Configuration validation passed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.

        Parameters
        ----------
        filepath : str
            Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SimulationConfig":
        """
        Load configuration from JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON file

        Returns
        -------
        SimulationConfig
            Loaded configuration
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        # Convert lists to tuples
        if "lon_range" in data and data["lon_range"] is not None:
            data["lon_range"] = tuple(data["lon_range"])
        if "lat_range" in data and data["lat_range"] is not None:
            data["lat_range"] = tuple(data["lat_range"])
        if "bc_lower" in data:
            data["bc_lower"] = tuple(data["bc_lower"])
        if "bc_upper" in data:
            data["bc_upper"] = tuple(data["bc_upper"])

        logger.info(f"Configuration loaded from {filepath}")
        return cls(**data)

    def __str__(self) -> str:
        """String representation of configuration."""
        lines = ["SimulationConfig:"]
        lines.append(f"  Domain: lon={self.lon_range}, lat={self.lat_range}")
        lines.append(f"  Grid: nx={self.nx}, ny={self.ny}")
        lines.append(f"  Time: t_final={self.t_final}s, dt={self.dt}s")
        lines.append(f"  Gravity: {self.gravity} m/s²")
        lines.append(f"  Output: {self.output_dir}")
        return "\n".join(lines)
