#!/usr/bin/env python3
"""
Setup script for the MFE Toolbox UI package.
This script works alongside pyproject.toml to provide both modern and
legacy Python package build support.

This setup file defines package metadata, dependencies, and build configuration
to enable installation via pip.
"""

import os
import sys
import re
from pathlib import Path
from setuptools import setup, find_packages  # setuptools >= 61.0.0
from typing import Dict, List, Optional, Union

# Directory containing this file
HERE = Path(__file__).resolve().parent

# Package name
PACKAGE_NAME = "mfe-toolbox-ui"

def get_version() -> str:
    """
    Extract the package version from mfe/ui/__init__.py using regular expressions.
    
    Returns:
        str: The package version string
    """
    init_path = HERE / "mfe" / "ui" / "__init__.py"
    if not init_path.exists():
        return "4.0.0"  # Default version if file not found
    
    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Look for line like: __version__ = "4.0.0"
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if version_match:
        return version_match.group(1)
    return "4.0.0"  # Default version if not found

def read_requirements() -> List[str]:
    """
    Read package requirements from requirements.txt file.
    
    Returns:
        List[str]: List of package dependencies with version specifiers
    """
    req_path = HERE / "requirements.txt"
    if not req_path.exists():
        # Core dependencies if requirements.txt not found
        return [
            "numpy>=1.26.3",
            "scipy>=1.11.4",
            "pandas>=2.1.4",
            "statsmodels>=0.14.1",
            "numba>=0.59.0",
            "PyQt6>=6.6.1",
            "matplotlib>=3.8.0"
        ]
    
    requirements = []
    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

def read_long_description() -> str:
    """
    Read the long description from README.md file.
    
    Returns:
        str: Content of README.md as string
    """
    readme_path = HERE / "README.md"
    if not readme_path.exists():
        return "MFE Toolbox UI - a comprehensive suite of Python modules for financial time series analysis and econometric modeling"
    
    with open(readme_path, "r", encoding="utf-8") as f:
        return f.read()

# Get the long description from README.md
LONG_DESCRIPTION = read_long_description()

# Get the list of requirements
REQUIREMENTS = read_requirements()

# Setup configuration
setup(
    name=PACKAGE_NAME,
    version=get_version(),
    description="The UI component for the MFE Toolbox - a comprehensive suite of Python modules for financial time series analysis and econometric modeling",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Kevin Sheppard",
    author_email="kevin.sheppard@economics.ox.ac.uk",
    packages=find_packages(include=["mfe", "mfe.*"]),
    install_requires=REQUIREMENTS,
    python_requires=">=3.12",
    entry_points={
        "console_scripts": [
            "mfe-ui = mfe.ui.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
        "Environment :: X11 Applications :: Qt"
    ],
    include_package_data=True,
)