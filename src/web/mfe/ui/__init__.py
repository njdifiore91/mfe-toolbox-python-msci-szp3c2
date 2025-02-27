"""
Initialization module for the MFE Toolbox's user interface package.
Exports key UI components, dialogs, and visualization tools for the PyQt6-based GUI, providing a clean API for accessing UI elements throughout the application.
"""
import logging  # standard library: Logging functionality for UI operations
import os  # standard library: Operating system interfaces for path handling
import typing  # standard library: Type hints for improved code documentation and type safety

from .main_window import MainWindow, run_application  # Main application window for the MFE Toolbox
from .armax_viewer import ARMAXViewer  # Dialog for viewing ARMAX model results
from .about_dialog import AboutDialog  # About dialog showing application information
from .close_dialog import CloseDialog  # Confirmation dialog for application closure
from .widgets import ARMAXWidget, ModelConfigurationWidget, DiagnosticVisualizationWidget, ControlButtonsWidget  # Main ARMAX model estimation widget

# Configure logger
logger = logging.getLogger(__name__)

__version__ = "4.0.0"

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
    "setup_ui_logging",
    "initialize_resources"
]


def setup_ui_logging(log_level: str) -> None:
    """
    Configures logging for the UI components.

    Args:
        log_level: Logging level string (e.g., "DEBUG", "INFO", "WARNING", "ERROR").

    Returns:
        None: No return value.
    """
    # Configure logging format and level for UI operations
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=numeric_level, format=log_format)

    # Set up file handlers for logging
    file_handler = logging.FileHandler("mfe_ui.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    # Configure console output for development environments
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # Log the UI initialization
    logger.info("UI logging initialized")


def initialize_resources() -> None:
    """
    Initializes UI resources like icons, styles, and plot configurations.

    Returns:
        None: No return value.
    """
    # Initialize matplotlib backend for PyQt6 integration
    import matplotlib
    matplotlib.use('Qt6Agg')

    # Load icon resources and application styles
    # (Implementation depends on the specific resource loading mechanism)
    logger.info("UI resources initialized")

    # Configure default plot styles and parameters
    # (Implementation depends on the specific plotting library)
    logger.info("Plot styles configured")

    # Initialize any runtime resources needed by the UI
    # (e.g., database connections, API clients)
    logger.info("Runtime resources initialized")