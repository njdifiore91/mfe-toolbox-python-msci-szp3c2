#!/bin/bash
# MFE Toolbox Test Suite Runner
# 
# This script runs the test suite for the MFE Toolbox, covering both
# backend and UI components. It enforces coverage requirements,
# supports property-based testing, and can run memory usage tests.
#
# The MFE Toolbox requires Python 3.12 for modern language features
# like async/await patterns and strict type hints. It also leverages
# Numba 0.59.0+ for performance-critical operations.
#
# Usage:
#   ./run_tests.sh [--memory] [pytest_args...]
#
# Options:
#   --memory    Run memory usage tests using pytest-memray
#   pytest_args Additional arguments to pass to pytest

# Directory setup
SCRIPT_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")
BACKEND_DIR=$ROOT_DIR/src/backend
WEB_DIR=$ROOT_DIR/src/web

# Test configuration
DEFAULT_COVERAGE=90
MIN_COVERAGE=${COVERAGE_THRESHOLD:-$DEFAULT_COVERAGE}
PYTHON_REQUIRED_VERSION="3.12"

# Parse arguments
RUN_MEMORY_TESTS=0
PYTEST_ARGS=()

for arg in "$@"; do
    if [ "$arg" == "--memory" ]; then
        RUN_MEMORY_TESTS=1
    else
        PYTEST_ARGS+=("$arg")
    fi
done

# Function to print a section header
print_header() {
    echo "----------------------------------------------"
    echo "    ${1^^}"
    echo "----------------------------------------------"
}

# Check Python version
check_python_version() {
    print_header "Checking Python version"
    
    # Get Python version
    python_version=$(python --version 2>&1)
    version_number=$(echo "$python_version" | sed -n 's/Python \([0-9]*\.[0-9]*\)\..*/\1/p')
    
    # Check if it's Python 3.12 (required for modern language features)
    if [[ "$version_number" == "$PYTHON_REQUIRED_VERSION" ]]; then
        echo "✓ Using Python $version_number as required"
        return 0
    else
        echo "❌ ERROR: Python $PYTHON_REQUIRED_VERSION is required, but found $version_number"
        echo "   The MFE Toolbox requires Python $PYTHON_REQUIRED_VERSION for modern language features"
        echo "   including async/await patterns and strict type hints."
        return 1
    fi
}

# Run backend tests
run_backend_tests() {
    print_header "Running backend tests"
    
    # Change to backend directory
    cd "$BACKEND_DIR" || { echo "❌ ERROR: Backend directory not found"; return 1; }
    
    # Check for Numba availability (required for performance-critical routines)
    if ! python -c "import numba" &>/dev/null; then
        echo "❌ ERROR: Numba is not installed"
        echo "   The MFE Toolbox requires Numba for performance-critical routines."
        echo "   Install with: pip install numba>=0.59.0"
        return 1
    fi
    
    # Run the pytest suite with coverage reporting
    python -m pytest \
        --cov=mfe \
        --cov-report=term \
        --cov-report=html:coverage_html \
        -n auto \
        --hypothesis-show-statistics \
        "${PYTEST_ARGS[@]}"
    
    pytest_status=$?
    
    # Run numba-specific tests for performance-critical routines
    print_header "Running numba performance tests"
    python -m pytest \
        -k "numba or performance" \
        --numba-captured-errors=new-style \
        "${PYTEST_ARGS[@]}"
    
    numba_status=$?
    
    # Return non-zero if either test suite failed
    if [ $pytest_status -ne 0 ] || [ $numba_status -ne 0 ]; then
        return 1
    fi
    
    return 0
}

# Run web UI tests
run_web_tests() {
    print_header "Running Web UI tests"
    
    # Change to web directory
    cd "$WEB_DIR" || { echo "❌ ERROR: Web UI directory not found"; return 1; }
    
    # Check if PyQt6 is available (required for UI components)
    if python -c "import PyQt6" &>/dev/null; then
        echo "✓ PyQt6 is available, running UI tests"
        
        # Run the pytest suite with coverage reporting
        python -m pytest \
            --cov=mfe.ui \
            --cov-report=term \
            --cov-report=html:coverage_html \
            -n auto \
            "${PYTEST_ARGS[@]}"
        
        return $?
    else
        echo "⚠ WARNING: PyQt6 is not available, skipping UI tests"
        echo "   The MFE Toolbox UI components require PyQt6."
        echo "   Install with: pip install PyQt6>=6.6.1"
        return 0
    fi
}

# Check if coverage meets requirements
check_coverage() {
    component_dir="$1"
    component_name="$2"
    coverage_file="$component_dir/.coverage"
    
    if [ ! -f "$coverage_file" ]; then
        echo "❌ ERROR: Coverage file not found: $coverage_file"
        return 1
    fi
    
    # Change to component directory to parse coverage report
    cd "$component_dir" || return 1
    
    # Generate coverage report to parse
    coverage_report=$(python -m coverage report)
    
    # Extract total coverage percentage
    coverage=$(echo "$coverage_report" | grep "TOTAL" | awk '{print $NF}' | tr -d "%")
    
    if [ -z "$coverage" ]; then
        echo "❌ ERROR: Could not extract coverage percentage from report"
        return 1
    fi
    
    # Check against threshold
    if [ "$coverage" -lt "$MIN_COVERAGE" ]; then
        echo "❌ ERROR: $component_name coverage is $coverage%, which is below the minimum threshold of $MIN_COVERAGE%"
        echo "   See coverage report in $component_dir/coverage_html/index.html"
        return 1
    else
        echo "✓ $component_name coverage is $coverage%, which meets the minimum threshold of $MIN_COVERAGE%"
        return 0
    fi
}

# Run memory usage tests
run_memory_tests() {
    print_header "Running memory usage tests"
    
    # Check if pytest-memray is installed
    if ! python -c "import pytest_memray" &>/dev/null; then
        echo "❌ ERROR: pytest-memray is not installed"
        echo "   Install with: pip install pytest-memray"
        return 1
    fi
    
    # Change to backend directory
    cd "$BACKEND_DIR" || { echo "❌ ERROR: Backend directory not found"; return 1; }
    
    # Run memory tests on selected test modules
    python -m pytest \
        --memray \
        --memray-bin-path=memray_reports \
        -k "test_models or test_core" \
        "${PYTEST_ARGS[@]}"
    
    return $?
}

# Main function
main() {
    # Check Python version first
    check_python_version
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Run backend tests
    run_backend_tests
    backend_status=$?
    
    # Check backend coverage
    check_coverage "$BACKEND_DIR" "Backend"
    backend_coverage_status=$?
    
    # Run web UI tests
    run_web_tests
    web_status=$?
    
    # Check web coverage only if UI tests were run
    web_coverage_status=0
    if [ -f "$WEB_DIR/.coverage" ]; then
        check_coverage "$WEB_DIR" "Web UI"
        web_coverage_status=$?
    fi
    
    # Run memory tests if requested
    memory_status=0
    if [ $RUN_MEMORY_TESTS -eq 1 ]; then
        run_memory_tests
        memory_status=$?
    fi
    
    # Print summary
    print_header "Test Summary"
    
    # Backend tests
    if [ $backend_status -eq 0 ]; then
        echo "✓ Backend tests: PASSED"
    else
        echo "❌ Backend tests: FAILED"
    fi
    
    # Backend coverage
    if [ $backend_coverage_status -eq 0 ]; then
        echo "✓ Backend coverage: MET"
    else
        echo "❌ Backend coverage: NOT MET"
    fi
    
    # Web UI tests
    if [ $web_status -eq 0 ]; then
        echo "✓ Web UI tests: PASSED"
    else
        echo "❌ Web UI tests: FAILED"
    fi
    
    # Web UI coverage
    if [ -f "$WEB_DIR/.coverage" ]; then
        if [ $web_coverage_status -eq 0 ]; then
            echo "✓ Web UI coverage: MET"
        else
            echo "❌ Web UI coverage: NOT MET"
        fi
    else
        echo "ℹ Web UI coverage: NOT CHECKED (tests skipped)"
    fi
    
    # Memory tests if applicable
    if [ $RUN_MEMORY_TESTS -eq 1 ]; then
        if [ $memory_status -eq 0 ]; then
            echo "✓ Memory tests: PASSED"
        else
            echo "❌ Memory tests: FAILED"
        fi
    fi
    
    # Determine overall status
    if [ $backend_status -eq 0 ] && [ $backend_coverage_status -eq 0 ] && [ $web_status -eq 0 ] && [ $web_coverage_status -eq 0 ] && [ $memory_status -eq 0 ]; then
        echo "✅ ALL TESTS PASSED"
        return 0
    else
        echo "❌ SOME TESTS FAILED"
        return 1
    fi
}

# Execute main function and pass its exit code to the script
main
exit $?