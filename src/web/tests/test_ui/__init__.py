"""
PyQt6 UI test package for MFE Toolbox.

This package provides a comprehensive testing framework for the PyQt6-based user interface
components of the MFE Toolbox. It organizes tests into logical modules that correspond to
UI components, model views, plotting functionality, and asynchronous operations.

The package leverages pytest's dynamic test discovery mechanism and integrates with 
Python's async/await patterns to enable effective testing of both synchronous and 
asynchronous UI behaviors.

Test Organization:
- test_widgets: Basic UI widget tests
- test_latex_renderer: LaTeX equation rendering tests
- test_plot_widgets: Matplotlib integration tests
- test_dialogs: Modal dialog tests
- test_armax_viewer: Results viewer component tests
- test_main_window: Main application window tests
- test_async: Asynchronous UI operation tests
- test_components: Reusable UI component tests
- test_models: Model visualization tests
- test_plots: Plot generation and interaction tests
"""

# Import pytest for test discovery and execution
import pytest  # pytest 7.4.3

# Import test subpackages
from . import test_async
from . import test_components
from . import test_models
from . import test_plots

# Define the modules exposed at the package level for test discovery
__all__ = [
    "test_widgets",
    "test_latex_renderer",
    "test_plot_widgets", 
    "test_dialogs",
    "test_armax_viewer",
    "test_main_window",
    "test_async",
    "test_components",
    "test_models",
    "test_plots"
]