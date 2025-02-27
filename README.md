# MFE Toolbox

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/mfe.svg)](https://badge.fury.io/py/mfe)
[![Documentation Status](https://readthedocs.org/projects/mfe-toolbox/badge/?version=latest)](https://mfe-toolbox.readthedocs.io/en/latest/?badge=latest)

## Project Overview

The MFE Toolbox is a comprehensive suite of Python modules designed for modeling financial time series and conducting advanced econometric analyses. It provides researchers, analysts, and practitioners with robust tools for:

- Financial time series modeling and forecasting
- Volatility and risk modeling using univariate and multivariate approaches  
- High-frequency financial data analysis
- Cross-sectional econometric analysis
- Bootstrap-based statistical inference
- Advanced distribution modeling and simulation

This toolbox leverages Python's scientific computing ecosystem, built upon foundational libraries including NumPy, SciPy, Pandas, and Statsmodels.

## Features

### Core Modules
- **Statistical Analysis**: Bootstrap methods, cross-sectional regression, distribution modeling, and statistical testing.
- **Time Series & Volatility**: ARMA/ARMAX models, univariate and multivariate volatility models, high-frequency analysis.

### Time Series & Volatility
- **ARMA/ARMAX Models**: Comprehensive framework for time series prediction and analysis.
- **Univariate Volatility**: Single-asset volatility models (AGARCH, APARCH, etc.).
- **Multivariate Volatility**: Multi-asset volatility models (BEKK, CCC, DCC).
- **High-Frequency Analysis**: Realized volatility estimation and noise filtering for intraday data.

### Performance Optimization
- **Numba JIT Compilation**: Just-in-time compilation for performance-critical functions.
- **Vectorized Operations**: Efficient NumPy array operations for large datasets.
- **Asynchronous Operations**: Improved performance in I/O-bound tasks.

### UI Components
- **Interactive Modeling Environment**: GUI interface built with PyQt6.
- **Diagnostic Plots**: Comprehensive visualization of model diagnostics.
- **Parameter Tables**: Clear display of model parameter estimates.

## Installation

### Prerequisites
- Python (Version 3.12)
- pip

### Pip Installation
```bash
pip install mfe
```

### Development Installation
```bash
git clone https://github.com/your-username/mfe-toolbox.git
cd mfe-toolbox
pip install -e .
```

## Quick Start

### Basic Import
```python
import mfe
```

### ARMA Example
```python
import numpy as np
from mfe.models import ARMA

# Generate sample data
np.random.seed(0)
data = np.random.randn(100)

# Create and estimate ARMA model
model = ARMA(p=1, q=1)
results = model.estimate(data)

# Print results summary
print(results.summary())
```

### GARCH Example
```python
import numpy as np
from mfe.models import GARCH

# Generate sample data
np.random.seed(0)
data = np.random.randn(100)

# Create and estimate GARCH model
model = GARCH(p=1, q=1)
results = model.fit(data)

# Print results summary
print(results.summary())
```

### Using Numba Optimization
```python
from numba import jit
import numpy as np

@jit(nopython=True)
def example_function(x, y):
    return x + y

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = example_function(a, b)
print(result)
```

## Package Structure

The MFE Toolbox is organized into the following namespaces:

### Core Modules
Fundamental statistical and computational components including bootstrap methods, distribution functions, optimization routines, and statistical tests.

### Models Module
Time series and volatility modeling implementations including ARMA/ARMAX models, univariate and multivariate volatility models, and high-frequency analysis tools.

### UI Module
User interface components for interactive modeling and visualization built with PyQt6.

### Utils Module
Utility functions and helper routines for data handling, validation, and performance optimization.

## Documentation

### Documentation Link
For detailed documentation, tutorials, and examples, please visit: [https://mfe-toolbox.readthedocs.io](https://mfe-toolbox.readthedocs.io)

### Tutorials
- Time Series Analysis with ARMA Models
- Volatility Modeling with GARCH
- High-Frequency Data Analysis

### Examples
- Basic ARMA Model Estimation
- GARCH Volatility Forecasting
- Realized Volatility Calculation

## Development

### Setup Environment
```bash
git clone https://github.com/your-username/mfe-toolbox.git
cd mfe-toolbox
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest
```

### Contributing Link
For information on contributing to the project, please see [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

### License Information
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgements

### Original Author
- Kevin Sheppard

### Contributors
- [CHANGELOG.md](./CHANGELOG.md)