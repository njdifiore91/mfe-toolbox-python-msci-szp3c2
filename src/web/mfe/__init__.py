"""
Initialization module for the web frontend of the MFE Toolbox. Provides package-level imports, version information, and exports UI components for the PyQt6-based graphical interface. Serves as the primary entry point for using the MFE Toolbox's web-based user interface components.
"""
import os  # standard library: Operating system interfaces for path handling
import sys  # standard library: System-specific parameters and functions
import logging  # standard library: Flexible event logging system for applications
import typing  # standard library: Support for type hints

from .ui.main_window import MainWindow, run_application  # Main application window for the MFE Toolbox
from .ui.armax_viewer import ARMAXViewer  # Dialog for viewing ARMAX model results
from .ui.about_dialog import AboutDialog  # About dialog showing application information
from .ui.close_dialog import CloseDialog  # Confirmation dialog for application closure
from .ui.widgets import ARMAXWidget, ModelConfigurationWidget, DiagnosticVisualizationWidget, ControlButtonsWidget  # UI widgets for model configuration and visualization
from .ui import setup_ui_logging, initialize_resources  # UI initialization functions

# Configure logger
logger = logging.getLogger(__name__)

__version__ = "4.0.0"
__author__ = "Kevin Sheppard"
__email__ = "kevin.sheppard@economics.ox.ac.uk"

__all__ = [
    "MainWindow",
    "ARMAXViewer",
    "AboutDialog",
    "CloseDialog",
    "ARMAXWidget",
    "ModelConfigurationWidget",
    "DiagnosticVisualizationWidget",
    "ControlButtonsWidget",
    "run_application",
    "initialize_ui",
    "get_version"
]


def initialize_ui(debug_mode: bool) -> bool:
    """
    Initializes the UI components and resources for the MFE Toolbox

    Args:
        debug_mode: A boolean indicating whether to run in debug mode.

    Returns:
        Initialization success status
    """
    try:
        # Configure logging with appropriate level based on debug_mode
        if debug_mode:
            log_level = "DEBUG"
        else:
            log_level = "INFO"
        setup_ui_logging(log_level)

        # Initialize UI resources like icons, styles, and plot configurations
        initialize_resources()

        # Set up the environment for PyQt6 operations
        # (e.g., setting the application style, loading plugins)
        logger.info("PyQt6 environment setup complete")

        # Log initialization status and return success flag
        logger.info("MFE Toolbox UI initialized successfully")
        return True
    except Exception as e:
        logger.error(f"MFE Toolbox UI initialization failed: {e}")
        return False


def get_version() -> str:
    """
    Returns the current version of the MFE Toolbox web frontend

    Returns:
        Version string in semantic versioning format
    """
    return __version__