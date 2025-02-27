# Contributing to MFE Toolbox

## Table of Contents
- [Introduction](#introduction)
- [Code of Conduct](#code-of-conduct)
- [Development Environment Setup](#development-environment-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Documentation Standards](#documentation-standards)
- [Release Process](#release-process)

## Introduction

Thank you for your interest in contributing to the MFE (Financial Econometrics) Toolbox! This document provides guidelines and instructions for contributing to the project. The MFE Toolbox is a Python-based suite of modules for financial time series analysis and econometric modeling, designed to provide robust tools for researchers and practitioners.

We welcome contributions in many forms, including:
- Bug reports and fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations
- Testing improvements

## Code of Conduct

This project adheres to a Code of Conduct that establishes expected behavior in our community. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Development Environment Setup

### Prerequisites
- Python 3.12
- Git

### Setting Up Your Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/mfe-toolbox.git
   cd mfe-toolbox
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

   This will install all required dependencies:
   - **Core Dependencies**:
     - NumPy (1.26.3+)
     - SciPy (1.11.4+)
     - Pandas (2.1.4+)
     - Statsmodels (0.14.1+)
     - Numba (0.59.0+)
     - PyQt6 (6.6.1+)
   
   - **Development Dependencies**:
     - pytest
     - pytest-cov
     - pytest-asyncio
     - hypothesis
     - flake8
     - mypy
     - black
     - isort
     - sphinx

5. **Verify your setup**
   ```bash
   pytest
   ```

## Development Workflow

### Branching Strategy

We follow a simplified Git workflow:

1. **Main branch**: Production-ready code
2. **Development branch**: Integration branch for features
3. **Feature branches**: Individual feature development

### Creating a Feature Branch

```bash
# Update your local repository
git checkout development
git pull origin development

# Create a feature branch
git checkout -b feature/your-feature-name
```

### Commit Guidelines

- Use clear, descriptive commit messages
- Begin with a capitalized, imperative verb (e.g., "Add", "Fix", "Update")
- Reference issue numbers when applicable
- Keep commits focused on a single logical change
- Example: `Add Numba optimization to GARCH estimation (#123)`

### Keeping Your Branch Updated

```bash
git checkout development
git pull origin development
git checkout feature/your-feature-name
git rebase development
```

## Coding Standards

### Style Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting

### Type Hints

- Use strict type hints for all functions and classes
- Leverage Python's typing module for complex types
- Example:
  ```python
  from typing import List, Optional, Tuple, Union
  import numpy as np
  
  def calculate_volatility(
      returns: np.ndarray,
      alpha: float = 0.05,
      window: Optional[int] = None
  ) -> Tuple[np.ndarray, float]:
      """
      Calculate volatility from return series.
      
      Parameters
      ----------
      returns : np.ndarray
          Array of return values
      alpha : float, optional
          Significance level, by default 0.05
      window : Optional[int], optional
          Rolling window size, by default None
          
      Returns
      -------
      Tuple[np.ndarray, float]
          Volatility series and mean volatility
      """
      # Implementation
  ```

### Modern Python Features

- Use dataclasses for parameter containers
- Implement async/await patterns for long-running operations
- Leverage context managers where appropriate
- Use Numba's @jit decorator for performance-critical functions

### Docstrings

- Follow NumPy/SciPy docstring format
- Include Parameters, Returns, Examples, and Notes sections
- Add references to academic papers where applicable

## Testing Guidelines

### Test Framework

- Use pytest as the primary test framework
- Use hypothesis for property-based testing
- Use numba.testing for performance validation

### Test Requirements

- All new features must include tests
- Aim for at least 90% test coverage
- Include unit tests, integration tests, and edge cases
- For statistical functions, verify against known results

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=mfe --cov-report=term-missing

# Run specific test file
pytest tests/test_models/test_garch.py

# Run only tests that match a pattern
pytest -k "garch"
```

### Performance Testing

- Include benchmarks for performance-critical functions
- Verify Numba optimizations with proper test cases
- Use pytest-benchmark for comparing implementations

## Pull Request Process

1. **Create a pull request** from your feature branch to the development branch
2. **Fill out the PR template** with:
   - Summary of changes
   - Related issue numbers
   - Testing approach
   - Documentation updates
3. **Pass all CI checks**:
   - Linting (flake8)
   - Type checking (mypy)
   - Tests (pytest)
   - Coverage thresholds
4. **Obtain code review** from at least one maintainer
5. **Address review comments** and update PR as needed
6. **Wait for approval** and merge by a maintainer

## Documentation Standards

### Code Documentation

- Document all public classes, methods, and functions
- Include docstrings for all modules
- Add inline comments for complex implementation details
- Update relevant examples when changing functionality

### Project Documentation

- Update README.md with new features or changes
- Keep API documentation current
- Add or update tutorials for significant features
- Document any breaking changes prominently

### Building Documentation

```bash
cd docs
make html
```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Functionality added in a backward compatible manner
- **PATCH**: Backward compatible bug fixes

### Release Preparation

1. Update version number in:
   - __init__.py
   - pyproject.toml
   - documentation
2. Update CHANGELOG.md
3. Create a release candidate for testing
4. Address any issues found during RC testing

### Creating a Release

Once approved, the maintainers will:
1. Merge to main branch
2. Tag the release
3. Build and publish to PyPI
4. Create GitHub release with release notes

---

Thank you for contributing to the MFE Toolbox. Your efforts help make this project better for everyone!