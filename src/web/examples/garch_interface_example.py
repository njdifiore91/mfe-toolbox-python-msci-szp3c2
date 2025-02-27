#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating how to use the PyQt6-based GARCH interface in the MFE Toolbox,
showcasing integration between backend models and visual UI components for interactive volatility modeling.
"""

import sys  # standard library
import asyncio  # standard library
import logging  # standard library
import argparse  # standard library
import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import matplotlib.pyplot as plt  # matplotlib 3.7.1
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton  # PyQt6 6.6.1
from PyQt6.QtWidgets import QSplitter, QLabel, QSizePolicy, QFrame  # PyQt6 6.6.1
from PyQt6.QtCore import Qt, pyqtSlot  # PyQt6 6.6.1

# Internal imports
from ...backend.mfe.models.garch import GARCH  # Core GARCH model for volatility estimation
from ..mfe.ui.models.garch_view import GARCHView  # UI component for GARCH model visualization and interaction
from ..mfe.ui/plots/volatility_plot import VolatilityPlot  # Specialized plot for volatility visualization
from ..mfe.ui.async/worker import WorkerManager  # Manages asynchronous task execution for UI responsiveness
from ...backend.mfe.utils.data_handling import load_financial_data  # Utility for loading financial datasets

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Constants
EXAMPLE_DATA_PATH = '../../backend/tests/test_data/market_benchmark.npy'
APP_TITLE = 'GARCH Model Interface Example'


def load_example_data(data_path: str) -> pd.Series:
    """
    Loads example financial data for GARCH modeling

    Args:
        data_path: Path to the data file

    Returns:
        pandas.Series: Log returns of example financial data
    """
    # Load financial data using load_financial_data function with specified path
    price_data = load_financial_data(data_path)

    # Calculate percentage returns from the price data
    returns = price_data.pct_change().dropna()

    # Convert to pandas Series with datetime index
    returns_series = pd.Series(returns, index=price_data.index[1:])

    # Return pandas Series of returns
    return returns_series


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configure the command-line argument parser for the example application

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    # Create ArgumentParser instance with description
    parser = argparse.ArgumentParser(description='PyQt6 GARCH Interface Example')

    # Add --data-path argument for custom data location
    parser.add_argument('--data-path', type=str, default=EXAMPLE_DATA_PATH,
                        help='Path to the financial data file')

    # Add --save-plot argument for saving volatility plots
    parser.add_argument('--save-plot', action='store_true',
                        help='Enable auto-saving of volatility plots')

    # Add --debug argument for enabling debug logging
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    # Return the configured parser
    return parser


async def async_run_garch_example(returns: pd.Series, model_params: dict) -> dict:
    """
    Asynchronously run GARCH model estimation on example data

    Args:
        returns: Time series of returns
        model_params: Dictionary of model parameters

    Returns:
        dict: Dictionary containing model, result, and forecast
    """
    # Create GARCH model instance with specified parameters
    model = GARCH(p=model_params['p'], q=model_params['q'])

    # Await model.fit_async with returns data
    result = await model.fit_async(returns)

    # Generate model summary containing parameters and statistics
    summary = model.summary()

    # Create volatility forecast for 20 periods ahead
    forecast = model.forecast(20)

    # Return dictionary with model, result, and forecast
    return {'model': model, 'result': result, 'forecast': forecast}


class GARCHInterfaceWindow(QMainWindow):
    """
    Main window for the GARCH interface example application
    """

    def __init__(self, returns_data: pd.Series, save_plots: bool):
        """
        Initialize the main application window with data
        """
        super().__init__()

        # Store returns_data and save_plots flag
        self._returns = returns_data
        self._save_plots = save_plots

        # Initialize UI components with setupUi method
        self.setupUi()

        # Create WorkerManager instance for async operations
        self._worker_manager = WorkerManager()

        # Set window title and size
        self.setWindowTitle(APP_TITLE)
        self.resize(900, 600)

        # Set up signal connections
        self.setupConnections()

        # Load data into GARCH view component
        self._garch_view.set_data(self._returns)

        # Log application window initialization
        logger.info('Application window initialized')

    def setupUi(self):
        """
        Set up the user interface components
        """
        # Create central widget and main layout
        self._central_widget = QWidget(self)
        self.setCentralWidget(self._central_widget)
        self._main_layout = QVBoxLayout(self._central_widget)

        # Create QSplitter for resizable panel separation
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create GARCHView widget and add to left panel
        self._garch_view = GARCHView()
        splitter.addWidget(self._garch_view)

        # Create VolatilityPlot widget and add to right panel
        self._volatility_plot = VolatilityPlot()
        splitter.addWidget(self._volatility_plot)

        # Create button panel with estimate, save plot, and quit buttons
        self._button_layout = QHBoxLayout()
        self._estimate_button = QPushButton('Estimate Model')
        self._save_plot_button = QPushButton('Save Plot')
        self._quit_button = QPushButton('Quit')
        self._save_plot_button.setEnabled(False)  # Initially disabled

        self._button_layout.addWidget(self._estimate_button)
        self._button_layout.addWidget(self._save_plot_button)
        self._button_layout.addWidget(self._quit_button)

        # Add all components to main layout
        self._main_layout.addWidget(splitter)
        self._main_layout.addLayout(self._button_layout)

        # Set central widget with main layout
        self._central_widget.setLayout(self._main_layout)

        # Initially disable save plot button
        self._save_plot_button.setEnabled(False)

    def setupConnections(self):
        """
        Set up signal-slot connections for UI components
        """
        # Connect estimate_button clicked to on_estimate_clicked
        self._estimate_button.clicked.connect(self.on_estimate_clicked)

        # Connect save_plot_button clicked to on_save_plot_clicked
        self._save_plot_button.clicked.connect(self.on_save_plot_clicked)

        # Connect quit_button clicked to close application
        self._quit_button.clicked.connect(self.close)

        # Connect GARCHView signals to corresponding slots
        # (Assuming GARCHView has signals for estimation progress, completion, and errors)
        # self._garch_view.estimation_progress.connect(self.on_estimation_progress)
        # self._garch_view.estimation_complete.connect(self.on_estimation_complete)
        # self._garch_view.estimation_error.connect(self.on_estimation_error)
        pass

    @pyqtSlot()
    def on_estimate_clicked(self):
        """
        Handle estimate button click by starting GARCH estimation
        """
        # Disable estimate button during estimation
        self._estimate_button.setEnabled(False)

        # Create async worker for GARCH model estimation
        worker = self._worker_manager.create_async_worker(self.async_estimate_model)
        worker.signals.result.connect(self.on_estimation_complete)
        worker.signals.error.connect(self.on_estimation_error)

        # Start worker for asynchronous execution
        self._worker_manager.start_worker(worker)
        logger.info('Estimation started')

    @pyqtSlot()
    def on_save_plot_clicked(self):
        """
        Handle save plot button click by saving volatility plots
        """
        # Check if GARCH view has results
        if not self._garch_view.has_results():
            QMessageBox.warning(self, 'Save Error', 'No results to save. Estimate model first.')
            return

        # Save volatility plot to file with timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f'volatility_plot_{timestamp}.png'
        self._volatility_plot.save_figure(filename)

        # Show success message
        QMessageBox.information(self, 'Save Successful', f'Plot saved to {filename}')

        # Log plot save operation
        logger.info(f'Volatility plot saved to {filename}')

    @pyqtSlot(object)
    def on_estimation_complete(self, result):
        """
        Handle completion of model estimation
        """
        # Enable estimate button again
        self._estimate_button.setEnabled(True)

        # Extract model, result, and forecast from result dict
        model = result['model']
        estimation_result = result['result']
        forecast = result['forecast']

        # Update GARCHView with estimation results
        self._garch_view.on_estimation_complete(estimation_result)

        # Update VolatilityPlot with returns and volatility data
        self._volatility_plot.set_data(self._returns, model.conditional_volatility, forecast)

        # Enable save plot button
        self._save_plot_button.setEnabled(True)

        # Show success message in status bar
        self.statusBar().showMessage('Estimation complete', 5000)

        # Auto-save plots if save_plots flag is True
        if self._save_plots:
            self.on_save_plot_clicked()

        # Log estimation completion
        logger.info('Estimation complete')

    @pyqtSlot(Exception)
    def on_estimation_error(self, error):
        """
        Handle errors during model estimation
        """
        # Enable estimate button again
        self._estimate_button.setEnabled(True)

        # Show error message dialog with error details
        QMessageBox.critical(self, 'Estimation Error', f'Failed to estimate model: {str(error)}')

        # Log error with traceback information
        logger.error(f'Estimation error: {str(error)}')

    async def async_estimate_model(self):
        """
        Asynchronously estimate GARCH model
        """
        # Define model parameters
        model_params = {'p': 1, 'q': 1}

        # Await async_run_garch_example with returns data and model parameters
        result = await async_run_garch_example(self._returns, model_params)

        # Return estimation result dictionary
        return result


def main():
    """
    Main entry point for the GARCH interface example application
    """
    # Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Configure debug logging if enabled
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Debug logging enabled')

    # Create application instance
    app = QApplication(sys.argv)

    # Load example data
    returns_data = load_example_data(args.data_path)

    # Create main window with data and save_plot flag
    window = GARCHInterfaceWindow(returns_data, args.save_plot)
    window.show()

    # Execute application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()