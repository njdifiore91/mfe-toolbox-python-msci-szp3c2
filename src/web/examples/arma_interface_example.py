#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
An example script demonstrating how to use the ARMA (AutoRegressive Moving Average)
model interface in the MFE Toolbox, showcasing the integration of the PyQt6-based
UI components with the backend ARMA model implementation for time series analysis.
"""

# Import necessary modules
import sys  # Access to Python interpreter variables and functions # Python 3.12
import os  # OS-dependent functionality for file path handling # Python 3.12
import asyncio  # Asynchronous I/O, event loop, and coroutines # Python 3.12
import logging  # Logging system for application events and errors # Python 3.12

import numpy as np  # Numerical operations for time series data manipulation # numpy 1.26.3
import pandas as pd  # Data manipulation and time series handling # pandas 2.1.4
from PyQt6.QtWidgets import (  # PyQt6 components for building the GUI # PyQt6.QtWidgets 6.6.1
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QWidget
)
from PyQt6.QtCore import Qt  # Core Qt functionality for UI application # PyQt6.QtCore 6.6.1
import matplotlib.pyplot as plt  # For creating and saving time series plots # matplotlib 3.8.0

# Internal imports
from src.web.mfe.ui.models.arma_view import ARMAView, ARMAViewSettings  # PyQt6 widget for ARMA model configuration and visualization
from src.backend.mfe.models.arma import ARMAModel  # Backend implementation of the ARMA model for time series analysis
from src.backend.mfe.utils.data_handling import load_data  # Utility function to load time series data for analysis

# Configure logger
logger = logging.getLogger(__name__)

# Global constants
EXAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../backend/tests/test_data/market_benchmark.npy')
DEFAULT_WINDOW_SIZE = (800, 600)

def load_example_data() -> pd.DataFrame:
    """
    Loads sample time series data for demonstration
    
    Returns:
        Time series data loaded as a Pandas DataFrame
    """
    try:
        # Check if example data file exists
        if not os.path.exists(EXAMPLE_DATA_PATH):
            raise FileNotFoundError(f"Example data file not found: {EXAMPLE_DATA_PATH}")
        
        # Load data from EXAMPLE_DATA_PATH using numpy.load
        data = np.load(EXAMPLE_DATA_PATH)
        
        # Convert the NumPy array to a Pandas DataFrame with date index
        df = pd.DataFrame(data, index=pd.date_range(start='2023-01-01', periods=len(data), freq='D'), columns=['Example Data'])
        
        # Return the DataFrame
        return df
    except FileNotFoundError as e:
        logger.error(f"Error loading example data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading example data: {e}")
        raise

def configure_logging():
    """
    Configures the basic logging setup for the example
    """
    # Configure logging format with timestamp and level
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log application startup message
    logger.info("Starting ARMA Interface Example")

def save_results_to_file(model: ARMAModel, filepath: str) -> bool:
    """
    Saves the ARMA model results to a file
    
    Args:
        model: ARMA model
        filepath: Filepath
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if model exists and has been estimated
        if model is None:
            logger.warning("Cannot export results: No model estimated")
            return False
        
        # Generate summary DataFrame from model
        summary_df = model.summary()
        
        # Save summary to CSV file
        summary_df.to_csv(filepath + "_summary.csv")
        
        # Save residual plot and forecast plot to PNG files
        # TODO: Implement residual and forecast plots
        
        # Return success status
        return True
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return False

class ARMAInterfaceExample(QMainWindow):
    """
    Main window class demonstrating the ARMA model interface within a PyQt6 application
    """
    
    def __init__(self):
        """
        Initialize the ARMA interface example window with UI components
        """
        # Call parent QMainWindow constructor
        super().__init__()
        
        # Set window title to 'MFE Toolbox - ARMA Interface Example'
        self.setWindowTitle("MFE Toolbox - ARMA Interface Example")
        
        # Set window size from DEFAULT_WINDOW_SIZE
        self.resize(DEFAULT_WINDOW_SIZE[0], DEFAULT_WINDOW_SIZE[1])
        
        # Initialize data to None
        self.data = None
        
        # Setup UI components through setup_ui method
        self.setup_ui()
        
        # Connect signal handlers to buttons
        self.load_data_button.clicked.connect(self.on_load_data_clicked)
        self.export_button.clicked.connect(self.on_export_clicked)
        self.forecast_button.clicked.connect(self.on_forecast_clicked)
        
        # Log application startup
        logger.info("ARMA Interface Example started")
    
    def setup_ui(self):
        """
        Sets up the UI components for the example window
        """
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create header section with title and description
        title_label = QLabel("ARMA Model Estimation")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        description_label = QLabel("Load data, configure ARMA model, and estimate parameters.")
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(description_label)
        
        # Create ARMA model view section with ARMAView widget
        self.arma_view = ARMAView()
        main_layout.addWidget(self.arma_view)
        
        # Create control section with load data, forecast, and export buttons
        control_layout = QHBoxLayout()
        self.load_data_button = QPushButton("Load Data")
        self.forecast_button = QPushButton("Forecast")
        self.export_button = QPushButton("Export")
        self.forecast_button.setEnabled(False)
        self.export_button.setEnabled(False)
        control_layout.addWidget(self.load_data_button)
        control_layout.addWidget(self.forecast_button)
        control_layout.addWidget(self.export_button)
        main_layout.addLayout(control_layout)
        
        # Create data information section with data_info_label
        self.data_info_label = QLabel("No data loaded")
        self.data_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.data_info_label)
        
        # Arrange all components in the main layout
        self.setCentralWidget(central_widget)
    
    def on_load_data_clicked(self):
        """
        Handler for load data button click
        """
        try:
            # Show file dialog for selecting a data file
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, "Load Data", "", "NumPy Files (*.npy);;CSV Files (*.csv);;All Files (*)")
            
            if file_path:
                # Process the loaded data
                if file_path.endswith('.npy'):
                    self.data = np.load(file_path)
                    self.data = pd.DataFrame(self.data)
                elif file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                else:
                    raise ValueError("Unsupported file format. Please load a .npy or .csv file.")
            else:
                # If dialog is canceled, attempt to load example data
                self.data = load_example_data()
            
            # Set the data in the ARMA view
            self.arma_view.set_data(self.data)
            
            # Update data_info_label with data information
            self.data_info_label.setText(f"Data loaded: {len(self.data)} observations")
            
            # Enable the forecast and export buttons
            self.forecast_button.setEnabled(True)
            self.export_button.setEnabled(True)
            
            # Log data loading success
            logger.info(f"Data loaded successfully from {file_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")
    
    def on_forecast_clicked(self):
        """
        Handler for forecast button click
        """
        try:
            # Check if a model has been estimated
            if self.arma_view.get_current_model() is None:
                QMessageBox.warning(self, "Warning", "Please estimate the model first.")
                return
            
            # Get the current model from ARMA view
            model = self.arma_view.get_current_model()
            
            # Show dialog for forecast horizon
            forecast_horizon, ok = QInputDialog.getInt(self, "Forecast Horizon", "Enter forecast horizon:", 10, 1, 1000)
            if not ok:
                return
            
            # Execute forecast calculation
            forecast_values = model.forecast(self.data['Example Data'].values, steps=forecast_horizon)
            
            # Display forecast results in a plot
            self.display_forecast_plot(self.data['Example Data'], forecast_values, np.zeros_like(forecast_values))
            
            # Log forecast generation
            logger.info(f"Forecast generated for {forecast_horizon} steps")
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate forecast: {e}")
    
    def on_export_clicked(self):
        """
        Handler for export button click
        """
        try:
            # Check if a model has been estimated
            if self.arma_view.get_current_model() is None:
                QMessageBox.warning(self, "Warning", "Please estimate the model first.")
                return
            
            # Get the current model from ARMA view
            model = self.arma_view.get_current_model()
            
            # Show file dialog for saving results
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(self, "Export Results", "", "CSV Files (*.csv);;All Files (*)")
            
            if file_path:
                # Call save_results_to_file function
                if save_results_to_file(model, file_path):
                    QMessageBox.information(self, "Success", f"Results exported to {file_path}")
                    logger.info(f"Results exported to {file_path}")
                else:
                    QMessageBox.critical(self, "Error", "Failed to export results.")
            
            # Log export operation
            logger.info("Export operation completed")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export results: {e}")
    
    def display_forecast_plot(self, actual_data: pd.Series, forecast_values: np.ndarray, forecast_variance: np.ndarray):
        """
        Displays a plot of forecast results
        
        Args:
            actual_data: Actual time series data
            forecast_values: Forecasted values
            forecast_variance: Forecast variance
        """
        try:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot actual data
            ax.plot(actual_data.index, actual_data.values, label="Actual Data")
            
            # Plot forecast values
            forecast_index = pd.date_range(start=actual_data.index[-1], periods=len(forecast_values) + 1, freq='D')[1:]
            ax.plot(forecast_index, forecast_values, label="Forecast", color='red')
            
            # Add confidence bands using forecast variance
            confidence_intervals = 1.96 * np.sqrt(forecast_variance)
            ax.fill_between(forecast_index, forecast_values - confidence_intervals, forecast_values + confidence_intervals, color='red', alpha=0.2, label="Confidence Interval")
            
            # Add labels and title
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.set_title("Time Series Forecast")
            ax.legend()
            
            # Display the plot window
            plt.show()
            
            # Log plot creation
            logger.info("Forecast plot created")
        except Exception as e:
            logger.error(f"Error displaying forecast plot: {e}")
            QMessageBox.critical(self, "Error", f"Failed to display forecast plot: {e}")
    
    def closeEvent(self, event):
        """
        Handler for window close events
        
        Args:
            event: Close event
        """
        # Accept the close event
        event.accept()
        
        # Log application shutdown
        logger.info("ARMA Interface Example shutting down")

def main():
    """
    Main entry point to run the ARMA interface example application
    """
    # Configure logging setup
    configure_logging()
    
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Create and show the ARMAInterfaceExample window
    window = ARMAInterfaceExample()
    window.show()
    
    # Execute the application main loop
    exit_code = app.exec()
    
    # Return application exit code
    return exit_code

if __name__ == "__main__":
    # Execute main function
    sys.exit(main())