# MFE Toolbox GUI

## Overview

The MFE (MATLAB Financial Econometrics) Toolbox GUI provides an interactive graphical interface for the Python-based MFE Toolbox (v4.0). Built with PyQt6, it offers a modern, cross-platform user experience for performing advanced econometric analyses while leveraging the computational power of the underlying Python libraries.

## Features

- **Interactive Model Configuration**: Easily set up ARMA/ARMAX, GARCH, and other econometric models through intuitive form controls
- **Real-time Visualization**: Dynamic plotting of model results, residuals, and diagnostic statistics
- **Asynchronous Processing**: Long-running calculations execute asynchronously to maintain UI responsiveness
- **Comprehensive Diagnostics**: Visual and statistical model validation tools
- **Result Export**: Save model results, plots, and statistics to various formats
- **Cross-platform Compatibility**: Works on Windows, macOS, and Linux systems
- **LaTeX Equation Rendering**: View mathematical model representations with publication-quality formatting

## Installation

### Prerequisites

- Python 3.12 or later
- MFE Toolbox core package
- PyQt6 (6.6.1 or later)

### Installation Methods

#### Via pip (recommended)

```bash
pip install mfe[gui]
```

This will install the MFE Toolbox with all GUI dependencies.

#### Manual Installation

If you have the MFE Toolbox already installed:

```bash
pip install pyqt6>=6.6.1 matplotlib>=3.7.0
```

## Usage

### Starting the GUI

```python
from mfe.ui import launch_gui

# Launch the main application window
launch_gui()
```

Alternatively, if installed via pip, you can use the command-line entry point:

```bash
mfe-gui
```

### Basic Workflow

1. **Configure Model**: Set up your model parameters using the input forms
2. **Load Data**: Import your time series data through the file dialog or paste from clipboard
3. **Estimate Model**: Click "Estimate Model" to begin the asynchronous computation
4. **View Results**: Explore results, diagnostics, and visualizations in the Results Viewer
5. **Export Results**: Save your analysis to various formats for reporting or further analysis

## Technical Details

### PyQt6 Implementation

The GUI is built using PyQt6, providing a native look and feel across all supported platforms. Key components include:

- Model configuration widgets with real-time validation
- Dynamic plot updates using the matplotlib QtAgg backend
- Asynchronous task handling to maintain UI responsiveness
- Custom LaTeX rendering for mathematical equations

### Asynchronous Processing

The GUI leverages Python's async/await patterns to handle computationally intensive tasks:

- Model estimation runs in background coroutines
- Real-time progress updates without blocking the UI
- Proper exception handling and error reporting
- Cancelable operations with clean resource management

### Cross-platform Support

The application is designed for consistent behavior across:

- Windows (x86_64)
- macOS (x86_64, arm64)
- Linux (x86_64)

## System Requirements

- **Python**: 3.12 or later
- **Memory**: 4GB RAM recommended for large datasets
- **Display**: 1280x720 or higher resolution
- **Dependencies**:
  - NumPy (1.26.3+)
  - SciPy (1.11.4+)
  - Pandas (2.1.4+)
  - Statsmodels (0.14.1+)
  - Numba (0.59.0+)
  - PyQt6 (6.6.1+)
  - Matplotlib (3.7.0+)

## Troubleshooting

### Common Issues

#### GUI Fails to Launch

- Ensure PyQt6 is properly installed: `pip install pyqt6`
- Check Python version compatibility: `python --version` (must be 3.12+)
- Verify all dependencies are installed: `pip check mfe`

#### Slow Performance

- For large datasets, increase available memory
- Enable Numba optimizations for faster computation
- Consider using the programmatic API for batch processing

#### Display Issues

- Update to the latest PyQt6 version: `pip install --upgrade pyqt6`
- Check for OS-specific display settings conflicts
- Try disabling high-DPI scaling if text appears blurry

## Contributing

Contributions to the MFE Toolbox GUI are welcome! Please refer to the main project repository for contribution guidelines.

## License

The MFE Toolbox GUI is released under the same license as the core MFE Toolbox.

## Acknowledgments

- Kevin Sheppard for the original MATLAB MFE Toolbox
- The PyQt team for the excellent Python Qt bindings
- The scientific Python community for the underlying computational libraries