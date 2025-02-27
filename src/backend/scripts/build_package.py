#!/usr/bin/env python3
"""
Build script for the MFE Toolbox Python package.

This script automates the build process for creating distributable package formats
(source distribution and wheel) using Python's modern packaging tools.
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path

# Global variables
PACKAGE_DIR = Path(__file__).parent.parent.resolve()
BUILD_TYPES = ['sdist', 'wheel']
DIST_DIR = PACKAGE_DIR / 'dist'
BUILD_DIR = PACKAGE_DIR / 'build'

# Add parent directory to path to import setup.py
sys.path.insert(0, str(PACKAGE_DIR))
try:
    from setup import get_version
except ImportError:
    print("Error: Could not import get_version from setup.py.")
    print(f"Current directory: {os.getcwd()}")
    print(f"Package directory: {PACKAGE_DIR}")
    sys.exit(1)

def parse_arguments():
    """
    Parse command-line arguments for build configuration.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Build script for MFE Toolbox Python package"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing build artifacts before building"
    )
    
    parser.add_argument(
        "--dist-type",
        choices=BUILD_TYPES + ["all"],
        default="all",
        help="Specify distribution type to build (sdist, wheel, or all)"
    )
    
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Skip dependency checks during build"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during build process"
    )
    
    return parser.parse_args()

def clean_build_artifacts(verbose=False):
    """
    Clean up existing build artifacts and directories.
    
    Args:
        verbose (bool): Whether to print detailed status messages
    """
    # Remove build directory if it exists
    if BUILD_DIR.exists():
        if verbose:
            print(f"Removing build directory: {BUILD_DIR}")
        shutil.rmtree(BUILD_DIR)
    
    # Remove dist directory if it exists
    if DIST_DIR.exists():
        if verbose:
            print(f"Removing dist directory: {DIST_DIR}")
        shutil.rmtree(DIST_DIR)
    
    # Remove any *.egg-info directories
    for egg_info in PACKAGE_DIR.glob("*.egg-info"):
        if verbose:
            print(f"Removing egg-info: {egg_info}")
        shutil.rmtree(egg_info)
    
    if verbose:
        print("Build artifacts cleaned successfully")

def build_distribution(dist_types, no_deps=False, verbose=False):
    """
    Build the specified distribution package(s).
    
    Args:
        dist_types (list): List of distribution types to build ('sdist', 'wheel', or both)
        no_deps (bool): Whether to skip dependency checks
        verbose (bool): Whether to print detailed status messages
    
    Returns:
        bool: Success status of the build process
    """
    os.chdir(PACKAGE_DIR)
    
    # Prepare build command
    build_cmd = [sys.executable, "-m", "build"]
    
    # Add distribution type options
    for dist_type in dist_types:
        if dist_type == "sdist":
            build_cmd.append("--sdist")
        elif dist_type == "wheel":
            build_cmd.append("--wheel")
    
    # Add no-deps flag if specified
    if no_deps:
        build_cmd.append("--no-deps")
    
    if verbose:
        print(f"Executing build command: {' '.join(build_cmd)}")
        print(f"Working directory: {PACKAGE_DIR}")
    
    # Execute build command
    try:
        result = subprocess.run(
            build_cmd,
            check=True,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None,
            text=True
        )
        
        # If not verbose but successful, show minimal output
        if not verbose:
            print(f"Successfully built MFE Toolbox v{get_version()} ({', '.join(dist_types)})")
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}")
        if not verbose and e.stdout:
            print("Build output:")
            print(e.stdout)
        if not verbose and e.stderr:
            print("Error output:")
            print(e.stderr)
        
        return False

def print_build_summary(success, dist_types):
    """
    Print a summary of the build results.
    
    Args:
        success (bool): Whether the build was successful
        dist_types (list): List of distribution types that were built
    """
    if not DIST_DIR.exists():
        print("No distribution files were created.")
        return
    
    # List all files in the dist directory
    dist_files = list(DIST_DIR.glob("*"))
    
    print("\nBuild Summary:")
    print("-" * 50)
    
    if success:
        print("✓ Build completed successfully")
    else:
        print("✗ Build failed")
    
    print(f"Distribution types: {', '.join(dist_types)}")
    
    if dist_files:
        print("\nCreated distribution files:")
        for file in dist_files:
            print(f"  - {file.name}")
    
    print("\nNext steps:")
    if success:
        if "wheel" in dist_types:
            print("  - Install locally: pip install dist/*.whl")
        else:
            print("  - Install locally: pip install dist/*.tar.gz")
        print("  - Upload to PyPI: python -m twine upload dist/*")
    else:
        print("  - Fix the build errors and try again")
    
    print("-" * 50)

def main():
    """
    Main entry point for the build script.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    args = parse_arguments()
    
    # Clean build artifacts if requested
    if args.clean:
        clean_build_artifacts(args.verbose)
    
    # Determine which distribution types to build
    if args.dist_type == "all":
        dist_types = BUILD_TYPES.copy()  # Make a copy to avoid modifying global
    else:
        dist_types = [args.dist_type]
    
    # Build the specified distribution package(s)
    success = build_distribution(
        dist_types=dist_types,
        no_deps=args.no_deps,
        verbose=args.verbose
    )
    
    # Print build summary
    print_build_summary(success, dist_types)
    
    # Return exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())