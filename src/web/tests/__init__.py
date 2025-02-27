"""
MFE Toolbox Web UI Test Package

This package provides testing infrastructure for the MFE Toolbox web user interface,
which is built using PyQt6. It includes test configuration, path setup utilities,
and test modules for UI components, plots, and asynchronous operations.

The package is designed to work with pytest as the primary test runner and supports
property-based testing via hypothesis where appropriate. It provides comprehensive
testing for PyQt6-based UI components and their integration with the core MFE
functionality.

The testing framework follows a modular structure aligned with the Python package
layout and provides specialized utilities for testing asynchronous UI operations
using Python's async/await patterns.
"""

import os
import logging
import pathlib
import pytest  # pytest 7.4.3

# Import UI test modules for test discovery
from . import test_ui

# Global constants for test configuration
TEST_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_UI_DIR = os.path.join(TEST_ROOT_DIR, 'test_ui')
TEST_LOGGER = logging.getLogger('mfe.web.tests')

# List of public API elements
__all__ = ['check_ui_test_environment', 'get_ui_test_path', 
           'setup_ui_test_environment', 'test_ui']


def check_ui_test_environment(verbose=False):
    """
    Verifies that the UI test environment is properly configured for MFE Toolbox web testing.
    
    This function checks for the presence of required packages (pytest, PyQt6) and
    confirms that the necessary test directories exist in the expected locations.
    
    Args:
        verbose (bool): If True, log detailed information about the environment status
        
    Returns:
        bool: True if the environment is properly configured, False otherwise
    """
    try:
        # Check for required packages
        import pytest
        
        # Verify PyQt6 is available for UI testing
        try:
            import PyQt6
            qt_available = True
        except ImportError:
            qt_available = False
            TEST_LOGGER.warning("PyQt6 is not available, UI tests will be skipped")
        
        # Check test directories exist
        ui_dir_exists = os.path.isdir(TEST_UI_DIR)
        
        if verbose:
            TEST_LOGGER.info(f"UI test environment check: PyQt6 available: {qt_available}")
            TEST_LOGGER.info(f"UI test environment check: UI test directory exists: {ui_dir_exists}")
        
        # All checks must pass for the environment to be considered properly configured
        return qt_available and ui_dir_exists
    
    except ImportError as e:
        if verbose:
            TEST_LOGGER.error(f"UI test environment check failed: {str(e)}")
        return False


def get_ui_test_path(path):
    """
    Returns the absolute path to a UI test file or directory.
    
    This utility function constructs an absolute path by joining the UI test root
    directory with the provided relative path. It validates that the resulting path
    exists to help identify configuration issues early.
    
    Args:
        path (str): Relative path to a test file or directory
        
    Returns:
        str: Absolute path to the UI test file or directory
    """
    # Construct absolute path to the requested test file or directory
    abs_path = os.path.join(TEST_UI_DIR, path)
    
    # Verify the path exists
    if not os.path.exists(abs_path):
        TEST_LOGGER.warning(f"UI test path does not exist: {abs_path}")
    
    return abs_path


def setup_ui_test_environment(verbose=False):
    """
    Sets up the UI test environment for the MFE Toolbox web tests.
    
    This function configures logging with appropriate verbosity and prepares
    the PyQt6 test environment. It should be called before running UI tests
    to ensure proper configuration.
    
    Args:
        verbose (bool): If True, enable verbose logging during test setup
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Verify environment configuration
    env_ok = check_ui_test_environment(verbose)
    
    if not env_ok:
        TEST_LOGGER.warning("UI test environment check failed - some UI tests may be skipped")
    
    try:
        # Setup PyQt6 test environment if available
        try:
            import PyQt6
            from PyQt6.QtWidgets import QApplication
            
            # Enable pytest to capture QApplication output for better test reporting
            app = QApplication.instance()
            if app is None:
                # Create application instance only if one doesn't exist
                # This prevents conflicts with tests that create their own QApplication
                app = QApplication([])
                
            if verbose:
                TEST_LOGGER.info("PyQt6 environment initialized successfully")
                
        except ImportError:
            if verbose:
                TEST_LOGGER.info("PyQt6 not available - UI tests will be skipped")
        
        if verbose:
            TEST_LOGGER.info("UI test environment setup completed")
        
        return True
    
    except Exception as e:
        TEST_LOGGER.error(f"UI test environment setup failed: {str(e)}")
        return False


# Run setup when script is executed directly
if __name__ == "__main__":
    setup_ui_test_environment(verbose=True)