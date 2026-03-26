"""Configuration dataclass for SWE Simulator."""

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from .config import SimulationConfig
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SWEResult:
    meshgrid_coord: tuple[np.ndarray | float, np.ndarray | float] = field(
        default_factory=lambda: (np.array([]), np.array([]))
    )
    meshgrid_metric: tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (np.array([]), np.array([]))
    )
    solution: np.ndarray = field(default_factory=lambda: np.array([]))
    bathymetry: np.ndarray = field(default_factory=lambda: np.array([]))
    initial_condition: np.ndarray = field(default_factory=lambda: np.array([]))
    wind_forcing: tuple[float | np.ndarray, float | np.ndarray] = field(
        default_factory=lambda: (0.0, 0.0)
    )
    config: SimulationConfig = field(default_factory=SimulationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, filepath: Path | str) -> None:
        logger.info(f"Saving SWe simulation result to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path | str) -> "SWEResult":
        logger.info(f"Loading SWE simulation result from {filepath}")
        with open(filepath, "rb") as f:
            return cast("SWEResult", pickle.load(f))
