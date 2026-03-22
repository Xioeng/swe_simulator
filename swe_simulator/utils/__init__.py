"""Utility subpackage exports for swe_simulator."""

from . import bathymetry, grid, io

try:
    from . import visualization
except ImportError:
    visualization = None

__all__ = [
    "bathymetry",
    "grid",
    "io",
    # "validation",
]

if visualization is not None:
    __all__.append("visualization")
