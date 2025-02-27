#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initialization module for the MFE Toolbox UI model views package that exports
various PyQt6-based view components for econometric models. This module
facilitates access to specialized UI components for different types of
financial models in a consistent interface.
"""

# Import necessary modules
import logging  # vstandard library
from typing import List  # vstandard library

# Internal imports
from .arma_view import ARMAView  # Access to ARMA model configuration and visualization
from .garch_view import GARCHView  # Access to GARCH model configuration and volatility visualization
from .distribution_view import DistributionView  # Access to statistical distribution analysis and visualization
from .univariate_view import UnivariateView  # Access to univariate volatility model visualization

# Configure logger
logger = logging.getLogger(__name__)

# Define the version of the UI models package
__version__ = "4.0.0"

# Define __all__ to specify what gets exported when using from mfe.ui.models import *
__all__: List[str] = ["ARMAView", "GARCHView", "DistributionView", "UnivariateView"]

logger.info(f"MFE Toolbox UI models package version {__version__} initialized")