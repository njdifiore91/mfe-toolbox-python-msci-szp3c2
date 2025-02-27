#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFE Toolbox web interface examples.

This package contains example modules demonstrating various aspects of the MFE Toolbox's
PyQt6-based user interface, including:

- LaTeX equation rendering
- Custom plot creation
- Asynchronous operations
- ARMA model interface
- GARCH model interface
- Basic GUI usage
- Responsive UI design

Each module can be run as a standalone script to demonstrate the specific functionality.
"""

__version__ = "0.1.0"
__author__ = "Kevin Sheppard"

from .latex_equation_example import LatexEquationExample  # Import LaTeX equation rendering example module
from .custom_plot_example import CustomPlotExample  # Import custom plotting example module
from .async_operation_example import AsyncOperationsExample  # Import asynchronous operations example module
from .arma_interface_example import ARMAInterfaceExample  # Import ARMA interface example module
from .garch_interface_example import GARCHInterfaceExample  # Import GARCH interface example module
from .basic_gui_usage import BasicGUIExample  # Import basic GUI usage example module
from .responsive_ui_example import ResponsiveUI  # Import responsive UI example module

__all__ = ["LatexEquationExample", "CustomPlotExample", "AsyncOperationsExample", "ARMAInterfaceExample", "GARCHInterfaceExample", "BasicGUIExample", "ResponsiveUI"]