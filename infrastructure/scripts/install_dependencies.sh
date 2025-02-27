#!/usr/bin/env bash
# install_dependencies.sh
# Script to install Python dependencies required by the MFE Toolbox

set -e  # Exit immediately if a command exits with a non-zero status

# Define function to get script directory (cross-platform)
get_script_dir() {
    local source="${BASH_SOURCE[0]}"
    # Resolve $source until the file is no longer a symlink
    while [ -L "$source" ]; do
        local dir="$( cd -P "$( dirname "$source" )" && pwd )"
        source="$(readlink "$source")"
        # If $source was a relative symlink, we need to resolve it relative to the path where the symlink file was located
        [[ $source != /* ]] && source="$dir/$source"
    done
    echo "$( cd -P "$( dirname "$source" )" && pwd )"
}

# Define global variables
SCRIPT_DIR=$(get_script_dir)
REPO_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
BACKEND_DIR=$REPO_ROOT/src/backend
WEB_DIR=$REPO_ROOT/src/web
PYTHON_VERSION="3.12"
VENV_NAME=".venv"

# Function to check if Python 3.12 is available
check_python_version() {
    echo "Checking Python version..."
    
    # Check if Python 3.12 is available
    if command -v python3.12 &>/dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3 &>/dev/null && [[ $(python3 -c "import sys; print(sys.version_info.minor)") -eq 12 ]]; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null && [[ $(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") == "3.12" ]]; then
        PYTHON_CMD="python"
    else
        echo "Error: Python 3.12 is required but not found."
        echo "Please install Python 3.12 from https://www.python.org/downloads/"
        echo "or use your system's package manager."
        return 1
    fi
    
    echo "Found Python 3.12: $(which $PYTHON_CMD)"
    return 0
}

# Function to set up a virtual environment
setup_virtual_environment() {
    local dir_path=$1
    
    echo "Setting up virtual environment in $dir_path..."
    cd "$dir_path" || return 2
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_NAME" ]; then
        echo "Creating new virtual environment..."
        $PYTHON_CMD -m venv "$VENV_NAME"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create virtual environment."
            return 2
        fi
    else
        echo "Using existing virtual environment."
    fi
    
    # Activate the virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        source "$VENV_NAME/Scripts/activate"
    else
        # Linux/macOS
        source "$VENV_NAME/bin/activate"
    fi
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    return 0
}

# Function to install backend dependencies
install_backend_dependencies() {
    echo "Installing backend dependencies..."
    cd "$BACKEND_DIR" || return 3
    
    # Install backend dependencies
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install backend dependencies."
            return 3
        fi
    else
        echo "Installing core scientific dependencies..."
        pip install numpy==1.26.3 scipy==1.11.4 pandas==2.1.4 statsmodels==0.14.1 numba==0.59.0
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install backend dependencies."
            return 3
        fi
    fi
    
    # Install the package in development mode
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        echo "Installing backend package in development mode..."
        pip install -e .
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install backend package."
            return 3
        fi
    fi
    
    echo "Backend dependencies installed successfully."
    return 0
}

# Function to install UI dependencies
install_web_dependencies() {
    echo "Installing UI dependencies..."
    cd "$WEB_DIR" || return 3
    
    # Install UI dependencies
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install UI dependencies."
            return 3
        fi
    else
        echo "Installing PyQt6..."
        pip install PyQt6==6.6.1
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install PyQt6."
            return 3
        fi
    fi
    
    # Install the package in development mode if setup.py exists
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        echo "Installing UI package in development mode..."
        pip install -e .
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install UI package."
            return 3
        fi
    fi
    
    echo "UI dependencies installed successfully."
    return 0
}

# Function to install development dependencies
install_dev_dependencies() {
    echo "Installing development dependencies..."
    cd "$BACKEND_DIR" || return 3
    
    # Install development dependencies
    if [ -f "requirements-dev.txt" ]; then
        echo "Installing development dependencies from requirements-dev.txt..."
        pip install -r requirements-dev.txt
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install development dependencies."
            return 3
        fi
    else
        echo "Installing pytest and related packages..."
        pip install pytest pytest-asyncio pytest-cov pytest-benchmark hypothesis mypy
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install pytest and related packages."
            return 3
        fi
    fi
    
    echo "Development dependencies installed successfully."
    return 0
}

# Main function
main() {
    echo "================================================"
    echo "MFE Toolbox Dependencies Installation"
    echo "================================================"
    echo "This script will install the following dependencies:"
    echo "- Python 3.12 (must be pre-installed)"
    echo "- Backend dependencies (NumPy, SciPy, Pandas, Statsmodels, Numba)"
    echo "- UI dependencies (PyQt6)"
    if [[ "$*" == *"--dev"* ]]; then
        echo "- Development dependencies (pytest, etc.)"
    fi
    echo "================================================"
    
    # Check Python version
    check_python_version
    if [ $? -ne 0 ]; then
        echo "Failed to verify Python 3.12 installation."
        exit 1
    fi
    
    # Setup virtual environment in backend directory
    setup_virtual_environment "$BACKEND_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to set up virtual environment."
        exit 2
    fi
    
    # Install backend dependencies
    install_backend_dependencies
    if [ $? -ne 0 ]; then
        echo "Failed to install backend dependencies."
        exit 3
    fi
    
    # Install UI dependencies
    install_web_dependencies
    if [ $? -ne 0 ]; then
        echo "Failed to install UI dependencies."
        exit 3
    fi
    
    # Install development dependencies if --dev flag is provided
    if [[ "$*" == *"--dev"* ]]; then
        install_dev_dependencies
        if [ $? -ne 0 ]; then
            echo "Failed to install development dependencies."
            exit 3
        fi
    fi
    
    echo "================================================"
    echo "Installation completed successfully!"
    echo ""
    echo "To activate the virtual environment, run:"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "  source $BACKEND_DIR/$VENV_NAME/Scripts/activate"
    else
        echo "  source $BACKEND_DIR/$VENV_NAME/bin/activate"
    fi
    echo "================================================"
    
    return 0
}

# Execute main function with all arguments
main "$@"