# MFE Toolbox: Financial Econometrics in Python

## Overview

The MFE (Financial Econometrics) Toolbox is a comprehensive suite of Python modules designed for modeling financial time series and conducting advanced econometric analyses. While retaining its legacy version 4.0 identity, the toolbox has been completely re-implemented using Python 3.12, incorporating modern programming constructs such as async/await patterns and strict type hints.

### Key Features

- Financial time series modeling and forecasting with ARMA/ARMAX
- Volatility and risk modeling using univariate (GARCH variants) and multivariate (BEKK, CCC, DCC) approaches
- High-frequency financial data analysis and realized volatility measures
- Cross-sectional econometric analysis
- Bootstrap-based statistical inference
- Advanced distribution modeling and simulation
- Numba-accelerated performance for computationally intensive operations
- Interactive PyQt6-based graphical user interface
- Comprehensive statistical testing framework

### Package Structure

The MFE Toolbox follows a modern Python package structure organized into four main namespaces:

```
mfe/
├── core/           # Core Statistical Modules
│   ├── bootstrap/
│   ├── distributions/
│   └── tests/
├── models/         # Time Series & Volatility Modules
│   ├── timeseries/
│   ├── univariate/
│   ├── multivariate/
│   └── realized/
├── ui/             # GUI Interface
│   └── widgets/
└── utils/          # Utility Functions
    └── performance/
```

## Installation

### Standard Installation

```bash
# Install from PyPI
pip install mfe
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/username/mfe-toolbox.git
cd mfe-toolbox

# Install in development mode
pip install -e .
```

### Requirements

- Python 3.12 or newer
- NumPy (1.26.3+)
- SciPy (1.11.4+)
- Pandas (2.1.4+)
- Statsmodels (0.14.1+)
- Numba (0.59.0+)
- PyQt6 (6.6.1+) [optional, for GUI components]

## Quick Start

### Basic Import

```python
import numpy as np
import mfe

# Check installed version
print(f"MFE Toolbox version: {mfe.__version__}")
```

### ARMA Example

```python
import numpy as np
from mfe.models.timeseries import ARMAX

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
data = 0.7 * np.roll(data, 1) + data
data = data[20:]  # Discard burn-in period

# Initialize and fit AR(1) model
model = ARMAX(ar_lags=[1], ma_lags=[], constant=True)
result = model.fit(data)

# Print results
print(result.summary())

# Generate forecasts
forecasts = model.forecast(result.params, steps=10)
print("Forecasts for next 10 periods:", forecasts)
```

### GARCH Example

```python
import numpy as np
from mfe.models.univariate import GARCH

# Generate sample returns
np.random.seed(42)
returns = np.random.normal(0, 1, 1000)

# Initialize and fit GARCH(1,1) model
model = GARCH(p=1, q=1)
result = model.fit(returns)

# Print results
print(result.summary())

# Forecast volatility
volatility_forecast = model.forecast_variance(result.params, steps=10)
print("Volatility forecasts for next 10 periods:", volatility_forecast)
```

## Core Components

### Core Statistical Modules

The `mfe.core` namespace contains fundamental statistical tools:

- **Bootstrap**: Robust resampling methods for dependent data series, including block bootstrap and stationary bootstrap implementations.
- **Distributions**: Advanced statistical distributions (GED, Hansen's skewed T) and related functions.
- **Tests**: Comprehensive statistical testing framework for model diagnostics and hypothesis testing.

### Time Series & Volatility

The `mfe.models` namespace provides time series and volatility modeling capabilities:

- **Timeseries**: ARMA/ARMAX modeling with comprehensive diagnostic tools and forecasting capabilities.
- **Univariate**: Single-asset volatility models (GARCH, EGARCH, APARCH, FIGARCH, etc.).
- **Multivariate**: Multi-asset volatility models (BEKK, CCC, DCC) for correlation and covariance forecasting.
- **Realized**: High-frequency financial econometrics with various realized volatility measures.

### Support Modules

The `mfe.utils` and `mfe.ui` namespaces provide supporting functionality:

- **UI**: Interactive modeling environment built with PyQt6 for visual analysis and model estimation.
- **Utils**: Data transformation, parameter validation, and helper functions.
- **Performance**: Numba-optimized computational kernels for performance-critical operations.

## Numba Optimization

### Performance Benefits

The MFE Toolbox utilizes Numba's just-in-time (JIT) compilation to accelerate performance-critical functions. This replaces the legacy MEX-based optimization in the original MATLAB implementation, providing:

- Near-C performance for numerical computations
- Automatic hardware-specific optimizations
- Seamless integration with the Python ecosystem
- Significant speedups for intensive calculations (typically 10-100x faster than pure Python)

### Optimization Example

```python
import numpy as np
from numba import jit
from mfe.core.optimization import numba_optimized_function

# Example of a Numba-optimized function
@jit(nopython=True)
def calculate_volatility(returns, alpha=0.94):
    """Calculate exponentially weighted moving average volatility"""
    n = len(returns)
    vols = np.zeros(n)
    vols[0] = returns[0] ** 2
    
    for i in range(1, n):
        vols[i] = alpha * vols[i-1] + (1 - alpha) * returns[i] ** 2
    
    return np.sqrt(vols)

# Usage
returns = np.random.normal(0, 1, 1000)
volatility = calculate_volatility(returns)
```

## Detailed Examples

### Included Examples

The package includes several detailed examples demonstrating key functionality:

- ARMA/ARMAX modeling and forecasting
- GARCH family model estimation and volatility forecasting
- Multivariate volatility models
- Bootstrap hypothesis testing
- High-frequency data analysis
- Cross-sectional regression
- Principal component analysis

### Example Usage

```python
# Import the examples module
from mfe import examples

# List available examples
print(examples.list_examples())

# Run a specific example
examples.run_example('arma_forecasting')
```

## Development

### Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=mfe

# Run specific test module
pytest tests/test_core/test_bootstrap.py
```

### Building Documentation

```bash
# Build Sphinx documentation
cd docs
make html

# View documentation
open _build/html/index.html  # On macOS
# Or: start _build/html/index.html  # On Windows
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

© 2023 Kevin Sheppard and contributors