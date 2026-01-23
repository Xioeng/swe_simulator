"""Custom exceptions for SWE Simulator."""


class SWESimulatorError(Exception):
    """Base exception for all SWE Simulator errors."""

    pass


class ConfigurationError(SWESimulatorError):
    """Raised when simulation configuration is invalid."""

    pass


class DomainNotSetError(ConfigurationError):
    """Raised when attempting to use solver before domain is configured."""

    def __init__(
        self, message: str = "Domain has not been configured. Call set_domain() first."
    ):
        super().__init__(message)


class BathymetryError(ConfigurationError):
    """Raised when there's an error with bathymetry data."""

    pass


class InitialConditionError(ConfigurationError):
    """Raised when there's an error with initial conditions."""

    pass


class TimeParametersError(ConfigurationError):
    """Raised when time parameters are invalid."""

    pass


class SolverError(SWESimulatorError):
    """Raised when the solver encounters an error during simulation."""

    pass


class ValidationError(SWESimulatorError):
    """Raised when input validation fails."""

    pass
