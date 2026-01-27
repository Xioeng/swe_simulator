"""Configuration dataclass for SWE Simulator."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import clawpack.petclaw as pyclaw

from .logging_config import get_logger

logger = get_logger(__name__)


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
    log_level : int, default=logging.INFO
        Logging level
    """

    # Domain parameters
    lon_range: Tuple[float, float] = None
    lat_range: Tuple[float, float] = None
    nx: int = 100
    ny: int = 100

    # Time parameters
    t_final: float = 10.0
    dt: float = 0.1

    # Physical parameters
    gravity: float = 9.81

    # Output parameters
    output_dir: str = "_output"
    frame_interval: int = 1
    multiple_output_times: bool = False

    # Numerical parameters
    cfl_desired: float = 0.9
    cfl_max: float = 1.0

    # Boundary conditions
    bc_lower: Tuple[pyclaw.BC, pyclaw.BC] = (pyclaw.BC.wall, pyclaw.BC.wall)
    bc_upper: Tuple[pyclaw.BC, pyclaw.BC] = (pyclaw.BC.wall, pyclaw.BC.wall)

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        self.validate()

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

        # Boundary conditions
        if len(self.bc_lower) != 2 or any(
            bc not in (pyclaw.BC.wall, pyclaw.BC.extrap, pyclaw.BC.periodic)
            for bc in self.bc_lower
        ):
            errors.append("bc_lower must be a tuple of (x, y) boundary conditions")

        if len(self.bc_upper) != 2 or any(
            bc not in (pyclaw.BC.wall, pyclaw.BC.extrap, pyclaw.BC.periodic)
            for bc in self.bc_upper
        ):
            errors.append("bc_upper must be a tuple of (x, y) boundary conditions")

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
        # Normalize types loaded from JSON
        for k in ("lon_range", "lat_range", "bc_lower", "bc_upper"):
            v = data.get(k)
            if v is not None:
                data[k] = tuple(v)

        for k in ("nx", "ny", "frame_interval"):
            if k in data and data[k] is not None:
                data[k] = int(data[k])

        if "multiple_output_times" in data:
            data["multiple_output_times"] = bool(data["multiple_output_times"])

        logger.info(f"Configuration loaded from {filepath}")
        return cls(**data)

    def __str__(self) -> str:
        """String representation of configuration."""
        lines = ["SimulationConfig:"]
        lines.append(f"  Domain: lon={self.lon_range}, lat={self.lat_range}")
        lines.append(f"  Grid: nx={self.nx}, ny={self.ny}")
        lines.append(f"  Time: t_final={self.t_final}s, dt={self.dt}s")
        lines.append(f"  Gravity: {self.gravity} m/s²")
        lines.append(f"  Output Directory: {self.output_dir}")
        lines.append(f"  Frame Interval: {self.frame_interval}")
        lines.append(f"  Multiple Output Times: {self.multiple_output_times}")
        lines.append(f"  CFL Desired: {self.cfl_desired}")
        lines.append(f"  CFL Max: {self.cfl_max}")
        lines.append(f"  Boundary Conditions Lower: {self.bc_lower}")
        lines.append(f"  Boundary Conditions Upper: {self.bc_upper}")
        return "\n".join(lines)
