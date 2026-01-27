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
    solution: np.ndarray
    bathymetry: np.ndarray
    initial_condition: np.ndarray
    wind_forcing: Union[np.ndarray, Tuple[float, float]]
    config: SimulationConfig

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: Path) -> None:
        logger.info(f"Saving SWe simulation result to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path) -> "SWEResult":
        logger.info(f"Loading SWe simulation result from {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)
