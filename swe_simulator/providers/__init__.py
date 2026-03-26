"""Providers for simulation inputs.

This module provides abstract base classes and concrete implementations
for supplying initial conditions, wind forcing, and bathymetry to the solver.

Classes
-------
InitialConditionProvider (ABC)
    Abstract base class for initial conditions

WindProvider (ABC)
    Abstract base class for wind forcing

BathymetryProvider (ABC)
    Abstract base class for bathymetry

GaussianHumpInitialCondition
    Gaussian hump elevation initial condition

FlatInitialCondition
    Flat rest-state initial condition

ConstantWind
    Time-independent constant wind

TimeVaryingWind
    Wind that varies with time via callables

FlatBathymetry
    Uniform depth bathymetry

SlopingBathymetry
    Linearly sloping bathymetry

Examples
--------
>>> from swe_simulator.providers import GaussianHumpInitialCondition, ConstantWind
>>> from swe_simulator.config import SimulationConfig
>>>
>>> config = SimulationConfig(nx=100, ny=100)
>>> ic = GaussianHumpInitialCondition(height=1.0)
>>> wind = ConstantWind(u_wind=5.0)
>>>
>>> initial_state = ic.get_initial_condition(config)
>>> wind_at_t = wind.get_wind(time=0.0)
"""

from .base import (
    BathymetryProvider,
    InitialConditionProvider,
    WindProvider,
)
from .bathymetry import (
    BathymetryFromCSV,
    BathymetryFromNC,
    FlatBathymetry,
    SlopingBathymetry,
)
from .initial_condition import (
    FlatInitialCondition,
    GaussianHumpInitialCondition,
    GaussianHumpInitialConditionNoGeo,
)
from .wind import ConstantWind

__all__ = [
    # Abstract base classes
    "InitialConditionProvider",
    "WindProvider",
    "BathymetryProvider",
    # Initial condition implementations
    "GaussianHumpInitialCondition",
    "GaussianHumpInitialConditionNoGeo",
    "FlatInitialCondition",
    # Wind implementations
    "ConstantWind",
    # "TimeVaryingWind",
    # Bathymetry implementations
    "FlatBathymetry",
    "SlopingBathymetry",
    "BathymetryFromNC",
    "BathymetryFromCSV",
]
