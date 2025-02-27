# MFE Toolbox User Interface Guide

## Introduction

This document provides a comprehensive guide to the user interface (UI) of the MFE (MATLAB Financial Econometrics) Toolbox. The MFE Toolbox features a modern, interactive UI built with PyQt6 <!-- PyQt6 6.6.1+ -->, designed to facilitate financial time series modeling and econometric analysis. This guide covers the key components, interaction patterns, and best practices for using the interactive modeling environment.

## Getting Started

To launch the MFE Toolbox GUI, ensure that you have installed the required dependencies, including Python 3.12, NumPy <!-- NumPy 1.26.3+ -->, SciPy <!-- SciPy 1.11.4+ -->, Pandas <!-- Pandas 2.1.4+ -->, Statsmodels <!-- Statsmodels 0.14.1+ -->, Numba <!-- Numba 0.59.0+ -->, and PyQt6 <!-- PyQt6 6.6.1+ -->. Then, execute the main script:

```python
from mfe.ui.main_window import MainWindow
import sys
from PyQt6.QtWidgets import QApplication

# Initialize Qt application
app = QApplication(sys.argv)

# Create and show the main window
main_window = MainWindow()
main_window.show()

# Start the application event loop
sys.exit(app.exec())
```

This will open the main application window, providing access to the toolbox's features.

## User Interface Overview

The MFE Toolbox UI is structured around a tabbed interface, providing access to different modeling and analysis tools. The main components include:

- **Menu Bar**: Provides access to file operations, tools, and help.
- **Toolbar**: Offers quick access to common actions.
- **Model Configuration Panel**: Allows users to specify model parameters.
- **Diagnostic Plots**: Displays diagnostic plots for model validation.
- **Results Viewer**: Presents detailed model estimation results.

![MFE Toolbox Main Window](screenshots/main_window.png)

## Main Application Window

The main application window is the primary interface for model configuration, estimation, and visualization. It includes the following key elements:

- **Model Configuration Panel**: Allows users to specify model parameters such as AR and MA orders, constant term options, and exogenous variables.
- **Diagnostic Plots**: Displays diagnostic plots such as residuals, ACF, and PACF for model validation.
- **Action Buttons**: Provides buttons for estimating the model, resetting parameters, viewing results, and closing the application.
- **Progress Indicator**: Shows the progress of long-running estimation tasks.

## Results Viewer

The Results Viewer is a modal dialog that presents comprehensive model estimation results. It includes the following key elements:

- **Model Equation**: Displays the estimated model equation using LaTeX rendering.
- **Parameter Estimates**: Presents a table of parameter estimates with standard errors, t-statistics, and p-values.
- **Statistical Metrics**: Shows key statistical metrics such as log-likelihood, AIC, and BIC.
- **Diagnostic Plots**: Displays diagnostic plots for model validation.
- **Navigation Controls**: Provides buttons for navigating between different result pages.

![MFE Toolbox Results Viewer](screenshots/results_viewer.png)

## Dialog Windows

The MFE Toolbox includes several secondary dialogs for specific tasks:

- **About Dialog**: Displays version and attribution details.
- **Close Confirmation Dialog**: Prompts the user to confirm closing the application with unsaved changes.

![MFE Toolbox About Dialog](screenshots/about_dialog.png)

## Model Configuration

The Model Configuration panel allows users to specify model parameters. Key elements include:

- **AR Order**: Specifies the order of the autoregressive (AR) component.
- **MA Order**: Specifies the order of the moving average (MA) component.
- **Include Constant**: Toggles the inclusion of a constant term in the model.
- **Exogenous Variables**: Selects exogenous variables to include in the model.

## Model Estimation

To estimate a model, follow these steps:

1. Configure model parameters in the Model Configuration panel.
2. Load data using the "Open" option in the File menu.
3. Click the "Estimate Model" button.
4. Monitor the progress of the estimation task using the progress indicator.
5. View the results in the Results Viewer dialog.

## Results Analysis

The Results Viewer dialog provides tools for interpreting model results:

- **Parameter Estimates**: Examine the parameter estimates and their statistical significance.
- **Statistical Metrics**: Compare information criteria (AIC, BIC) to assess model fit.
- **Diagnostic Plots**: Analyze residual plots, ACF, and PACF to validate model assumptions.

## Diagnostic Plots

The MFE Toolbox provides several diagnostic plots for model validation:

- **Residuals Plot**: Visualizes the distribution of residuals.
- **ACF Plot**: Displays the autocorrelation function of the residuals.
- **PACF Plot**: Displays the partial autocorrelation function of the residuals.

See [Plot Reference](plots.md) for more details.

## Exporting Results

To export estimation results, click the "Export Results" button in the Results Viewer dialog. This will save the results to a file in a specified format.

## Asynchronous Operations

Long-running operations such as model estimation are handled asynchronously to maintain UI responsiveness. A progress indicator is displayed during these operations, and cancellation options are provided.

See [Async Operations Reference](async_operations.md) for more details.

## Keyboard Shortcuts

The MFE Toolbox provides keyboard shortcuts for efficient interaction:

| Action | Shortcut |
|---|---|
| Open File | Ctrl+O |
| Save | Ctrl+S |
| Estimate Model | Ctrl+E |
| View Results | Ctrl+V |
| Close | Ctrl+W |
| About | Ctrl+H |

## Customization

The MFE Toolbox UI can be customized to suit user preferences. Options include:

- **Theme Selection**: Choose between light and dark themes.
- **Font Size**: Adjust the font size for improved readability.
- **Plot Styling**: Customize the appearance of diagnostic plots.

## Troubleshooting

Common issues and their solutions:

- **Error Messages**: Check the error message for details and consult the documentation.
- **Unexpected Behavior**: Restart the application and try again.
- **Performance Considerations**: Close unused result viewers to free resources.

## Advanced Usage

Advanced techniques for power users:

- **Integration with External Python Scripts**: Use the MFE Toolbox API to integrate with custom Python scripts.
- **Combining Multiple Analysis Procedures**: Chain multiple analysis procedures together using the MFE Toolbox API.

## Component Reference

See [UI Components in MFE Toolbox](components.md) for detailed documentation about individual UI components.

## Plot Reference

See [Plotting in MFE Toolbox](plots.md) for detailed documentation about plotting capabilities.

## Async Operations Reference

See [Asynchronous Operations in MFE Toolbox](async_operations.md) for detailed documentation about asynchronous operations.