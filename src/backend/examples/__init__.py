"""
Initialization module for the MFE Toolbox examples package.
Provides a centralized entry point for example scripts, exposes common
functionality, and facilitates access to the demonstration modules that
showcase the toolbox's capabilities with realistic financial econometric use cases.
"""

import os  # Python 3.12 - Access to operating system interfaces for path operations
import logging  # Python 3.12 - Logging functionality for example execution
from typing import Any, Dict  # Python 3.12 - Type hints for better code safety and documentation
from importlib import import_module  # Python 3.12 - Dynamic module import functionality
from types import ModuleType  # Python 3.12 - Type hinting for modules

__version__ = "1.0.0"
EXAMPLE_MODULES = [
    'data_handling',
    'bootstrap_analysis',
    'arma_modeling',
    'garch_volatility',
    'realized_measures',
    'multivariate_volatility',
    'custom_distribution'
]
EXAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
logger = logging.getLogger(__name__)
__all__ = ['run_example', 'list_examples', 'get_example_data_path', *EXAMPLE_MODULES]


def run_example(example_name: str, params: Dict[str, Any]) -> Any:
    """
    Runs a specified example module by name

    Parameters
    ----------
    example_name : str
        Name of the example module
    params : Dict[str, Any]
        Parameters to pass to the example module's run function

    Returns
    -------
    Any
        Result of the example execution if any
    """
    if example_name not in EXAMPLE_MODULES:
        raise ValueError(f"Invalid example name: {example_name}. Available examples: {EXAMPLE_MODULES}")

    logger.info(f"Starting example: {example_name}")

    # Dynamically import the example module
    module = load_example_module(example_name)

    # Call the module's run_example function with provided parameters
    result = module.run_data_handling_examples() if hasattr(module, 'run_data_handling_examples') else None

    logger.info(f"Completed example: {example_name}")
    return result


def list_examples() -> Dict[str, str]:
    """
    Returns a list of available example modules with descriptions

    Returns
    -------
    Dict[str, str]
        Dictionary mapping example names to their descriptions
    """
    example_descriptions = {}
    for module_name in EXAMPLE_MODULES:
        try:
            module = load_example_module(module_name)
            example_descriptions[module_name] = module.__doc__ or "No description available."
        except Exception as e:
            example_descriptions[module_name] = f"Error loading description: {str(e)}"
    return example_descriptions


def get_example_data_path(filename: str) -> str:
    """
    Returns the path to example data files

    Parameters
    ----------
    filename : str
        Name of the data file

    Returns
    -------
    str
        Absolute path to the example data file
    """
    if not filename:
        return EXAMPLE_DATA_PATH
    file_path = os.path.join(EXAMPLE_DATA_PATH, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return os.path.abspath(file_path)


def load_example_module(module_name: str) -> ModuleType:
    """
    Dynamically loads an example module by name

    Parameters
    ----------
    module_name : str
        Name of the module to load

    Returns
    -------
    ModuleType
        The loaded Python module
    """
    if module_name not in EXAMPLE_MODULES:
        raise ValueError(f"Invalid module name: {module_name}. Available modules: {EXAMPLE_MODULES}")
    module_path = f"src.backend.examples.{module_name}"
    return import_module(module_path)