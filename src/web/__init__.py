"""
Initialize the web package for the MFE Toolbox, providing access to the PyQt6-based user interface components. Serves as the entry point for integrating the web UI with the core backend functionality.
"""
import os  # Python 3.12 standard library: Operating system interfaces for path handling
import sys  # Python 3.12 standard library: System-specific parameters and functions
import logging  # Python 3.12 standard library: Flexible event logging system for applications
import typing  # Python 3.12 standard library: Support for type hints

from src.backend.mfe import initialize  # internal import: Initialize the MFE Toolbox backend environment
from .mfe import mfe  # internal import: Import the web MFE package with UI components
from .ui.main_window import MainWindow, run_application  # internal import: Main application window for the MFE Toolbox
from .ui.armax_viewer import ARMAXViewer  # internal import: Dialog for viewing ARMAX model results
from .ui.about_dialog import AboutDialog  # internal import: About dialog showing application information
from .ui.close_dialog import CloseDialog  # internal import: Confirmation dialog for application closure
from .ui.widgets import ARMAXWidget, ModelConfigurationWidget, DiagnosticVisualizationWidget, ControlButtonsWidget  # internal import: UI widgets for model configuration and visualization
from .ui import setup_ui_logging, initialize_resources  # internal import: UI initialization functions

# Configure logger
logger = logging.getLogger(__name__)

__version__ = "4.0.0"
__author__ = "Kevin Sheppard"
__email__ = "kevin.sheppard@economics.ox.ac.uk"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "MainWindow",
    "ARMAXViewer",
    "AboutDialog",
    "CloseDialog",
    "ARMAXWidget",
    "ModelConfigurationWidget",
    "DiagnosticVisualizationWidget",
    "ControlButtonsWidget",
    "run_application",
    "initialize_web",
    "get_version"
]


def initialize_web(quiet: bool, debug_mode: bool) -> bool:
    """
    Initializes both the MFE Toolbox backend and web UI environment

    Args:
        quiet: A boolean indicating whether to suppress output.
        debug_mode: A boolean indicating whether to run in debug mode.

    Returns:
        Success status of the initialization
    """
    try:
        # Configure logging based on debug_mode parameter
        setup_logging(debug_mode)

        # Initialize the backend MFE Toolbox using initialize() with quiet parameter
        backend_initialized = initialize(quiet=quiet)

        # Initialize the web UI components using mfe.initialize_ui() with debug_mode parameter
        ui_initialized = mfe.initialize_ui(debug_mode)

        # Log initialization status
        if backend_initialized and ui_initialized:
            logger.info("MFE Toolbox backend and web UI initialized successfully.")
        else:
            logger.warning("MFE Toolbox initialization partially failed.")

        # Return combined success status of backend and UI initialization
        return backend_initialized and ui_initialized
    except Exception as e:
        logger.error(f"MFE Toolbox initialization failed: {e}")
        return False


def get_version() -> str:
    """
    Returns the current version of the MFE Toolbox web package

    Returns:
        Version string in semantic versioning format
    """
    # Return the __version__ global variable
    return __version__


def setup_logging(debug_mode: bool) -> logging.Logger:
    """
    Set up logging configuration for the web package

    Args:
        debug_mode: A boolean indicating whether to enable debug mode.

    Returns:
        Configured logger for the package
    """
    # Set log level based on debug_mode (DEBUG if True, INFO otherwise)
    log_level = logging.DEBUG if debug_mode else logging.INFO

    # Configure root logger with appropriate level
    logging.basicConfig(level=log_level)

    # Get or create a logger using logging.getLogger(__name__)
    logger = logging.getLogger(__name__)

    # Configure a stream handler if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Return configured logger
    return logger