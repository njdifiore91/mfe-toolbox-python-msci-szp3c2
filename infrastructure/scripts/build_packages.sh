#!/bin/bash
# build_packages.sh - Build script for MFE Toolbox Python packages
# 
# This script automates the building of Python packages for both the MFE Toolbox
# backend and web components, creating source and wheel distributions for PyPI
# publishing or direct installation.
#
# The script ensures proper integration of Numba-optimized modules and handles
# the build configuration defined in pyproject.toml.

# Setup the environment for the build process
setup_environment() {
    # Exit on error
    set -e
    
    # Get the script directory
    SCRIPT_DIR="$(dirname "$0")"
    
    # Determine the repository root directory
    ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
    
    # Define paths to backend and web directories
    BACKEND_DIR="$ROOT_DIR/src/backend"
    WEB_DIR="$ROOT_DIR/src/web"
    
    # Create output directory for collected packages
    OUTPUT_DIR="$ROOT_DIR/dist"
    
    # Set default build flags (all enabled)
    BUILD_BACKEND=true
    BUILD_WEB=true
    BUILD_SDIST=true
    BUILD_WHEEL=true
    
    # Set Python command
    PYTHON_CMD="python3"
}

# Display usage information
print_usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo "Build Python packages for MFE Toolbox backend and web components."
    echo ""
    echo "Options:"
    echo "  --backend-only    Build only the backend package"
    echo "  --web-only        Build only the web UI package"
    echo "  --sdist-only      Build only source distributions"
    echo "  --wheel-only      Build only wheel distributions"
    echo "  --python=<cmd>    Specify Python command (default: python3)"
    echo "  --help            Display this help message"
    echo ""
    echo "Examples:"
    echo "  $(basename "$0") --backend-only   # Build only the backend package"
    echo "  $(basename "$0") --python=python3.12   # Use Python 3.12 for building"
}

# Parse command-line arguments
parse_arguments() {
    for arg in "$@"; do
        case $arg in
            --backend-only)
                BUILD_BACKEND=true
                BUILD_WEB=false
                shift
                ;;
            --web-only)
                BUILD_BACKEND=false
                BUILD_WEB=true
                shift
                ;;
            --sdist-only)
                BUILD_SDIST=true
                BUILD_WHEEL=false
                shift
                ;;
            --wheel-only)
                BUILD_SDIST=false
                BUILD_WHEEL=true
                shift
                ;;
            --python=*)
                PYTHON_CMD="${arg#*=}"
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo "Error: Unknown option: $arg" >&2
                print_usage
                exit 1
                ;;
        esac
    done
}

# Check for required build tools
check_build_tools() {
    echo "Checking for required build tools..."
    
    # Check if Python command exists
    if ! command -v $PYTHON_CMD &>/dev/null; then
        echo "Error: Python command '$PYTHON_CMD' not found." >&2
        echo "Please install Python or specify the correct command with --python=<cmd>" >&2
        return 1
    fi
    
    # Check Python version
    local python_version=$($PYTHON_CMD --version | awk '{print $2}')
    echo "Using Python $python_version ($PYTHON_CMD)"
    
    # Check if Python build module is installed
    if ! $PYTHON_CMD -c "import build" &>/dev/null; then
        echo "Installing Python build module..."
        $PYTHON_CMD -m pip install build
    fi
    
    # Check for wheel and setuptools
    if ! $PYTHON_CMD -c "import wheel, setuptools" &>/dev/null; then
        echo "Installing wheel and setuptools..."
        $PYTHON_CMD -m pip install wheel setuptools
    fi
    
    # Check for numba if building backend
    if [[ "$BUILD_BACKEND" == true ]]; then
        if ! $PYTHON_CMD -c "import numba" &>/dev/null; then
            echo "Warning: Numba is not installed in the current Python environment." >&2
            echo "This is required for performance-optimized modules." >&2
            echo "Install with: $PYTHON_CMD -m pip install numba" >&2
        else
            echo "Numba is available for optimized module compilation."
        fi
    fi
    
    echo "All required build tools are available."
    return 0
}

# Check if Numba-optimized modules are properly included
check_numba_modules() {
    local dir="$1"
    echo "Checking for Numba-optimized modules in $dir..."
    
    local numba_files=("core/optimization.py" "models/garch.py" "models/volatility.py" 
                       "core/bootstrap.py" "models/realized.py" "core/distributions.py")
    
    for file in "${numba_files[@]}"; do
        if [[ ! -f "$dir/mfe/$file" ]]; then
            echo "Warning: Numba-optimized module not found: mfe/$file" >&2
        else
            # Check if file contains @jit decorator
            if ! grep -q "@jit" "$dir/mfe/$file"; then
                echo "Warning: Numba @jit decorator not found in mfe/$file" >&2
            fi
        fi
    done
    
    echo "Numba module check completed."
}

# Verify pyproject.toml configuration
check_pyproject_toml() {
    local dir="$1"
    echo "Checking pyproject.toml configuration in $dir..."
    
    if [[ ! -f "$dir/pyproject.toml" ]]; then
        echo "Error: pyproject.toml not found in $dir" >&2
        return 1
    fi
    
    # Check for build-system section
    if ! grep -q "build-system" "$dir/pyproject.toml"; then
        echo "Error: build-system section missing in pyproject.toml" >&2
        return 1
    fi
    
    # Check for project section
    if ! grep -q "project" "$dir/pyproject.toml"; then
        echo "Error: project section missing in pyproject.toml" >&2
        return 1
    fi
    
    # Check for dependencies
    if [[ "$dir" == "$BACKEND_DIR" ]]; then
        # Check for numba dependency
        if ! grep -q "numba" "$dir/pyproject.toml"; then
            echo "Warning: numba dependency not specified in pyproject.toml" >&2
        fi
        
        # Check for other required dependencies
        for dep in "numpy" "scipy" "pandas" "statsmodels"; do
            if ! grep -q "$dep" "$dir/pyproject.toml"; then
                echo "Warning: $dep dependency not specified in pyproject.toml" >&2
            fi
        done
    fi
    
    echo "pyproject.toml configuration check completed."
    return 0
}

# Build the backend Python package
build_backend() {
    echo "Building backend package (mfe)..."
    
    # Check if backend directory exists
    if [[ ! -d "$BACKEND_DIR" ]]; then
        echo "Error: Backend directory not found: $BACKEND_DIR" >&2
        return 1
    fi
    
    # Check pyproject.toml configuration
    check_pyproject_toml "$BACKEND_DIR" || return 1
    
    # Check Numba-optimized modules
    check_numba_modules "$BACKEND_DIR"
    
    # Navigate to backend directory
    cd "$BACKEND_DIR"
    
    # Build source distribution if enabled
    if [[ "$BUILD_SDIST" == true ]]; then
        echo "Building source distribution..."
        $PYTHON_CMD -m build --sdist
    fi
    
    # Build wheel distribution if enabled
    if [[ "$BUILD_WHEEL" == true ]]; then
        echo "Building wheel distribution..."
        $PYTHON_CMD -m build --wheel
    fi
    
    # Return to original directory
    cd - > /dev/null
    
    echo "Backend package build completed."
    return 0
}

# Build the web UI Python package
build_web() {
    echo "Building web UI package..."
    
    # Check if web directory exists
    if [[ ! -d "$WEB_DIR" ]]; then
        echo "Error: Web directory not found: $WEB_DIR" >&2
        return 1
    fi
    
    # Check pyproject.toml configuration
    check_pyproject_toml "$WEB_DIR" || return 1
    
    # Navigate to web directory
    cd "$WEB_DIR"
    
    # Build source distribution if enabled
    if [[ "$BUILD_SDIST" == true ]]; then
        echo "Building source distribution..."
        $PYTHON_CMD -m build --sdist
    fi
    
    # Build wheel distribution if enabled
    if [[ "$BUILD_WHEEL" == true ]]; then
        echo "Building wheel distribution..."
        $PYTHON_CMD -m build --wheel
    fi
    
    # Return to original directory
    cd - > /dev/null
    
    echo "Web UI package build completed."
    return 0
}

# Collect built packages into a single directory
collect_packages() {
    echo "Collecting packages to $OUTPUT_DIR..."
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Copy backend packages if backend was built
    if [[ "$BUILD_BACKEND" == true ]]; then
        if [[ -d "$BACKEND_DIR/dist" ]]; then
            cp -f "$BACKEND_DIR/dist"/*.{whl,tar.gz} "$OUTPUT_DIR/" 2>/dev/null || true
        fi
    fi
    
    # Copy web packages if web was built
    if [[ "$BUILD_WEB" == true ]]; then
        if [[ -d "$WEB_DIR/dist" ]]; then
            cp -f "$WEB_DIR/dist"/*.{whl,tar.gz} "$OUTPUT_DIR/" 2>/dev/null || true
        fi
    fi
    
    # Ensure proper permissions
    chmod -R 644 "$OUTPUT_DIR"/*.{whl,tar.gz} 2>/dev/null || true
    
    echo "Package collection completed."
}

# Verify the built packages
verify_packages() {
    echo "Verifying built packages..."
    
    # Count the number of packages
    local package_count=$(find "$OUTPUT_DIR" -name "*.whl" -o -name "*.tar.gz" | wc -l)
    
    if [[ $package_count -eq 0 ]]; then
        echo "Error: No packages were built." >&2
        return 1
    fi
    
    # Print summary of built packages
    echo "Built packages:"
    find "$OUTPUT_DIR" -name "*.whl" -o -name "*.tar.gz" | sort | while read pkg; do
        echo "  - $(basename "$pkg")"
    done
    
    # Verify package filenames follow expected pattern
    if [[ "$BUILD_BACKEND" == true ]]; then
        if [[ "$BUILD_WHEEL" == true ]] && ! find "$OUTPUT_DIR" -name "mfe-*.whl" | grep -q .; then
            echo "Warning: No backend wheel package found." >&2
        fi
        
        if [[ "$BUILD_SDIST" == true ]] && ! find "$OUTPUT_DIR" -name "mfe-*.tar.gz" | grep -q .; then
            echo "Warning: No backend source package found." >&2
        fi
    fi
    
    if [[ "$BUILD_WEB" == true ]]; then
        if [[ "$BUILD_WHEEL" == true ]] && ! find "$OUTPUT_DIR" -name "mfe_web-*.whl" | grep -q .; then
            echo "Warning: No web wheel package found." >&2
        fi
        
        if [[ "$BUILD_SDIST" == true ]] && ! find "$OUTPUT_DIR" -name "mfe_web-*.tar.gz" | grep -q .; then
            echo "Warning: No web source package found." >&2
        fi
    fi
    
    # Check for cross-platform wheel tags
    if [[ "$BUILD_WHEEL" == true ]]; then
        # Check for pure Python wheels (which are cross-platform)
        if find "$OUTPUT_DIR" -name "*-py3-none-any.whl" | grep -q .; then
            echo "Found pure Python wheels (cross-platform compatible)."
        else
            echo "Warning: No pure Python wheels found. Platform-specific wheels may limit compatibility." >&2
        fi
    fi
    
    echo "Package verification completed. $package_count package(s) found."
    return 0
}

# Main function
main() {
    # Initialize the environment
    setup_environment
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Check for required build tools
    check_build_tools || return 1
    
    # Build packages
    local build_status=0
    
    if [[ "$BUILD_BACKEND" == true ]]; then
        build_backend || build_status=1
    fi
    
    if [[ "$BUILD_WEB" == true ]]; then
        build_web || build_status=1
    fi
    
    if [[ $build_status -ne 0 ]]; then
        echo "Error: Build process failed." >&2
        return $build_status
    fi
    
    # Collect packages
    collect_packages
    
    # Verify packages
    verify_packages || return 1
    
    echo "Build process completed successfully."
    echo "Packages are available in: $OUTPUT_DIR"
    
    return 0
}

# Execute main function
main "$@"
exit $?