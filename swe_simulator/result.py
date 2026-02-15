"""Configuration dataclass for SWE Simulator."""

import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np

from .config import SimulationConfig
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SWEResult:
    meshgrid_coord: Tuple[np.ndarray, np.ndarray] = None
    meshgrid_metric: Tuple[np.ndarray, np.ndarray] = None
    solution: np.ndarray = None
    bathymetry: np.ndarray = None
    initial_condition: np.ndarray = None
    wind_forcing: Union[np.ndarray, Tuple[float, float]] = None
    config: SimulationConfig = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: Path) -> None:
        logger.info(f"Saving SWe simulation result to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "SWEResult":
        logger.info(f"Loading SWE simulation result from {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)
