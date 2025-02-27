"""
Core initialization module for the MFE Toolbox.

This module provides environment setup, Python version compatibility checking,
path configuration, and Numba optimization verification. It serves as the entry
point for configuring the package environment and ensuring all dependencies are
properly loaded.
"""

import os
import sys
import logging
import warnings
import importlib
import platform
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path

# Internal imports
from .utils.numba_helpers import check_numba_compatibility
from .utils.validation import validate_params

# Configure module logger
logger = logging.getLogger(__name__)

# Define constants
REQUIRED_PACKAGES = [
    {'name': 'numpy', 'version': '1.26.3'},
    {'name': 'scipy', 'version': '1.11.4'},
    {'name': 'pandas', 'version': '2.1.4'},
    {'name': 'statsmodels', 'version': '0.14.1'},
    {'name': 'numba', 'version': '0.59.0'},
    {'name': 'pyqt6', 'version': '6.6.1'}
]

MINIMUM_PYTHON_VERSION = (3, 12)
PACKAGE_PATHS = ['core', 'models', 'ui', 'utils']
INITIALIZATION_COMPLETE = False


def initialize(quiet: bool = False, additional_paths: Optional[List[str]] = None) -> bool:
    """
    Main entry point for initializing the MFE Toolbox environment.
    
    Parameters
    ----------
    quiet : bool, default=False
        If True, suppresses non-essential output and sets log level to WARNING
    additional_paths : Optional[List[str]], default=None
        Additional paths to add to the Python path during initialization
        
    Returns
    -------
    bool
        True if initialization succeeds, False otherwise
    """
    global INITIALIZATION_COMPLETE
    
    # Validate input parameters
    param_specs = {
        'quiet': {'type': bool, 'required': False},
        'additional_paths': {'type': (list, type(None)), 'required': False}
    }
    
    params = {
        'quiet': quiet,
        'additional_paths': additional_paths
    }
    
    if not validate_params(params, param_specs):
        # Configure basic logging first for error reporting
        logging.basicConfig(level=logging.ERROR)
        logger.error("Invalid parameters provided to initialize")
        return False
    
    # Configure logging based on quiet parameter
    configure_logging(quiet)
    
    # Log initialization start
    logger.info("Initializing MFE Toolbox environment...")
    
    # Check Python version
    if not check_python_version():
        logger.error(f"Python version check failed. Minimum required version: {MINIMUM_PYTHON_VERSION}")
        return False
    
    # Check required packages
    package_status = check_required_packages()
    missing_packages = [pkg for pkg, status in package_status.items() if not status]
    if missing_packages:
        logger.error(f"Required package check failed. Missing or incompatible packages: {', '.join(missing_packages)}")
        return False
    
    # Setup path configuration
    added_paths = setup_path_configuration()
    if not added_paths:
        logger.error("Failed to set up path configuration")
        return False
    
    # Check Numba availability
    if not check_numba_availability():
        logger.warning("Numba optimization is not available. Performance will be degraded.")
    
    # Add additional paths if specified
    if additional_paths:
        for path in additional_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.append(path)
                logger.debug(f"Added additional path: {path}")
    
    # Mark initialization as complete
    INITIALIZATION_COMPLETE = True
    logger.info("MFE Toolbox initialization completed successfully")
    
    return True


def initialize_environment(
    quiet: bool = False,
    additional_paths: Optional[List[str]] = None,
    verify_packages: bool = True,
    verify_numba: bool = True
) -> bool:
    """
    Comprehensive environment setup for the MFE Toolbox.
    
    This function provides a more flexible interface for initialization with
    options to control package verification and Numba optimization checks.
    
    Parameters
    ----------
    quiet : bool, default=False
        If True, suppresses non-essential output
    additional_paths : Optional[List[str]], default=None
        Additional paths to add to the Python path
    verify_packages : bool, default=True
        If True, verifies required package availability and versions
    verify_numba : bool, default=True
        If True, verifies Numba JIT compilation functionality
        
    Returns
    -------
    bool
        True if environment setup succeeds, False otherwise
    """
    global INITIALIZATION_COMPLETE
    
    # Validate input parameters
    param_specs = {
        'quiet': {'type': bool, 'required': False},
        'additional_paths': {'type': (list, type(None)), 'required': False},
        'verify_packages': {'type': bool, 'required': False},
        'verify_numba': {'type': bool, 'required': False}
    }
    
    params = {
        'quiet': quiet,
        'additional_paths': additional_paths,
        'verify_packages': verify_packages,
        'verify_numba': verify_numba
    }
    
    if not validate_params(params, param_specs):
        logger.error("Invalid parameters provided to initialize_environment")
        return False
    
    # Configure logging
    configure_logging(quiet)
    
    # Check Python version (always required)
    if not check_python_version():
        logger.error(f"Python version check failed. Minimum required version: {MINIMUM_PYTHON_VERSION}")
        return False
    
    # Setup path configuration (always required)
    if not setup_path_configuration():
        logger.error("Failed to set up path configuration")
        return False
    
    # Optionally verify packages
    if verify_packages:
        package_status = check_required_packages()
        missing_packages = [pkg for pkg, status in package_status.items() if not status]
        if missing_packages:
            logger.error(f"Required package check failed. Missing or incompatible packages: {', '.join(missing_packages)}")
            return False
    
    # Optionally verify Numba
    if verify_numba:
        if not check_numba_availability():
            logger.warning("Numba optimization is not available. Performance will be degraded.")
    
    # Add additional paths if specified
    if additional_paths:
        for path in additional_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.append(path)
                logger.debug(f"Added additional path: {path}")
    
    # Mark initialization as complete
    INITIALIZATION_COMPLETE = True
    logger.info("MFE Toolbox environment setup completed successfully")
    
    return True


def check_python_version(minimum_version: Optional[Tuple[int, int]] = None) -> bool:
    """
    Verify that the Python runtime version meets the minimum requirements.
    
    Parameters
    ----------
    minimum_version : Optional[Tuple[int, int]], default=None
        Minimum required Python version as (major, minor) tuple.
        If None, uses MINIMUM_PYTHON_VERSION global.
        
    Returns
    -------
    bool
        True if Python version is compatible, False otherwise
    """
    if minimum_version is None:
        minimum_version = MINIMUM_PYTHON_VERSION
    
    current_version = sys.version_info[:2]
    
    if current_version < minimum_version:
        warnings.warn(
            f"Python version {'.'.join(map(str, current_version))} is not supported. "
            f"Minimum required version is {'.'.join(map(str, minimum_version))}."
        )
        logger.warning(
            f"Python version check failed. Current version: {'.'.join(map(str, current_version))}, "
            f"required: {'.'.join(map(str, minimum_version))}"
        )
        return False
    
    logger.info(f"Python version check passed. Current version: {'.'.join(map(str, current_version))}")
    return True


def check_required_packages(required_packages: Optional[List[Dict[str, str]]] = None) -> Dict[str, bool]:
    """
    Verify that all required packages are installed with compatible versions.
    
    Parameters
    ----------
    required_packages : Optional[List[Dict[str, str]]], default=None
        List of dictionaries with package requirements, each containing 'name' and 'version' keys.
        If None, uses REQUIRED_PACKAGES global.
        
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping package names to availability status (True if available and compatible)
    """
    if required_packages is None:
        required_packages = REQUIRED_PACKAGES
    
    results = {}
    
    for package_info in required_packages:
        package_name = package_info['name']
        required_version = package_info['version']
        
        # Check if package can be imported
        try:
            # Try to import the package
            module = importlib.import_module(package_name)
            
            # Get package version
            current_version = get_package_version(package_name)
            
            if current_version is None:
                logger.warning(f"Could not determine version for package: {package_name}")
                results[package_name] = False
                continue
            
            # Compare versions
            if not compare_versions(current_version, required_version):
                logger.warning(
                    f"Package {package_name} version {current_version} is not compatible. "
                    f"Required version: {required_version}"
                )
                results[package_name] = False
                continue
            
            logger.info(f"Package {package_name} version {current_version} is compatible")
            results[package_name] = True
            
        except ImportError:
            warnings.warn(f"Required package {package_name} is not installed")
            logger.warning(f"Package {package_name} is not installed")
            results[package_name] = False
    
    return results


def check_numba_availability() -> bool:
    """
    Verify that Numba is available and functioning correctly for JIT compilation.
    
    Returns
    -------
    bool
        True if Numba is available and working, False otherwise
    """
    try:
        # Try to import numba
        import numba
        
        # Define a simple test function
        def test_function(x, y):
            return x + y
        
        # Check if the function can be JIT-compiled
        if check_numba_compatibility(test_function):
            logger.info("Numba JIT compilation is working properly")
            return True
        else:
            logger.warning("Numba is installed but JIT compilation is not working")
            return False
    
    except ImportError:
        warnings.warn("Numba is not installed. Performance will be degraded.")
        logger.warning("Numba is not installed")
        return False


def setup_path_configuration(
    package_paths: Optional[List[str]] = None,
    base_path: Optional[str] = None
) -> List[str]:
    """
    Configure the Python path to include all necessary package directories.
    
    Parameters
    ----------
    package_paths : Optional[List[str]], default=None
        List of package directories to add to the Python path.
        If None, uses PACKAGE_PATHS global.
    base_path : Optional[str], default=None
        Base directory for package paths. If None, determined from current module.
        
    Returns
    -------
    List[str]
        List of paths successfully added to the Python path
    """
    # Validate input parameters
    param_specs = {
        'package_paths': {'type': (list, type(None)), 'required': False},
        'base_path': {'type': (str, Path, type(None)), 'required': False}
    }
    
    params = {
        'package_paths': package_paths,
        'base_path': base_path
    }
    
    if not validate_params(params, param_specs, raise_error=False):
        logger.error("Invalid parameters provided to setup_path_configuration")
        return []
    
    if package_paths is None:
        package_paths = PACKAGE_PATHS
    
    # Determine base path if not provided
    if base_path is None:
        # Get directory of current module
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    added_paths = []
    
    for directory in package_paths:
        # Construct full path
        if isinstance(base_path, Path):
            full_path = str(base_path / directory)
        else:
            full_path = os.path.join(base_path, directory)
        
        # Check if path exists
        if os.path.exists(full_path):
            # Add to Python path if not already present
            if full_path not in sys.path:
                sys.path.append(full_path)
                added_paths.append(full_path)
                logger.debug(f"Added path to Python path: {full_path}")
        else:
            logger.warning(f"Package directory not found: {full_path}")
    
    if added_paths:
        logger.info(f"Added {len(added_paths)} paths to Python path")
    else:
        logger.warning("No paths were added to Python path")
    
    return added_paths


def configure_logging(quiet: bool = False, level: Optional[int] = None) -> logging.Logger:
    """
    Configure the logging system for the MFE Toolbox.
    
    Parameters
    ----------
    quiet : bool, default=False
        If True, sets log level to WARNING, otherwise INFO
    level : Optional[int], default=None
        Explicit log level to use. If provided, overrides quiet parameter.
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Determine log level
    if level is None:
        level = logging.WARNING if quiet else logging.INFO
    
    # Get logger for the mfe package
    logger = logging.getLogger('mfe')
    
    # Configure logger if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Set log level
    logger.setLevel(level)
    
    return logger


def get_package_version(package_name: str) -> Optional[str]:
    """
    Get the current version of a package.
    
    Parameters
    ----------
    package_name : str
        Name of the package
        
    Returns
    -------
    Optional[str]
        Version string if package is installed, None otherwise
    """
    try:
        # Try to import the package
        module = importlib.import_module(package_name)
        
        # Try to get version from __version__ attribute
        if hasattr(module, '__version__'):
            return module.__version__
        
        # Try to get version from VERSION attribute
        if hasattr(module, 'VERSION'):
            version = module.VERSION
            if isinstance(version, tuple):
                return '.'.join(map(str, version))
            return str(version)
        
        # Try pkg_resources as a fallback
        try:
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except (ImportError, pkg_resources.DistributionNotFound):
            pass
        
        return None
    
    except ImportError:
        return None


def compare_versions(current_version: str, required_version: str) -> bool:
    """
    Compare two version strings for compatibility.
    
    Parameters
    ----------
    current_version : str
        Current version string
    required_version : str
        Required version string
        
    Returns
    -------
    bool
        True if current version meets or exceeds required version, False otherwise
    """
    def normalize_version(version_str):
        """Normalize version string for comparison."""
        # Split version into components
        components = []
        for part in version_str.split('.'):
            # Extract numeric part
            numeric = ''.join(c for c in part if c.isdigit())
            if numeric:
                components.append(int(numeric))
            else:
                components.append(0)
        
        # Ensure at least 3 components (major, minor, patch)
        while len(components) < 3:
            components.append(0)
        
        return components
    
    current = normalize_version(current_version)
    required = normalize_version(required_version)
    
    # Compare major, minor, patch versions
    for c, r in zip(current, required):
        if c > r:
            return True
        elif c < r:
            return False
    
    # If we get here, versions are equal or current has more components
    return True