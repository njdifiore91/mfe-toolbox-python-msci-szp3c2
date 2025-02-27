"""
MFE Toolbox - Models Package

This package provides a comprehensive suite of time series, volatility, and
high-frequency financial models implemented in Python 3.12 with Numba optimization.
"""

__version__ = "4.0.0"

# Time Series Models
from .arma import ARMAModel, ARMAResults  # Time series ARMA modeling
from .armax import ARMAXModel, ARMAXResults  # Time series ARMAX modeling with exogenous variables

# Volatility Models
from .volatility import (
    VolatilityModel,
    VolatilityResult,
    VolatilityType,
    VolatilityForecast,
    DistributionType,
    estimate_volatility,
    forecast_volatility,
    simulate_volatility,
)  # Base classes and utilities for volatility modeling
from .garch import GARCH  # GARCH volatility model

# Realized Volatility
from .realized import (
    realized_variance,
    realized_kernel,
    realized_volatility,
    realized_covariance,
    RealizedVolatility,
)  # Realized volatility measures

__all__ = [
    "ARMAModel",
    "ARMAResults",
    "ARMAXModel",
    "ARMAXResults",
    "VolatilityModel",
    "VolatilityResult",
    "VolatilityType",
    "VolatilityForecast",
    "DistributionType",
    "estimate_volatility",
    "forecast_volatility",
    "simulate_volatility",
    "GARCH",
    "realized_variance",
    "realized_kernel",
    "realized_volatility",
    "realized_covariance",
    "RealizedVolatility",
]