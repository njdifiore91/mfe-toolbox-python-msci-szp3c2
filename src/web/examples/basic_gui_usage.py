"""
A simple example script demonstrating basic usage of the MFE Toolbox GUI components,
focusing on creating and interacting with the main application window for financial econometric analysis.
"""
import sys  # Python standard library: System-specific parameters and functions
import os  # Python standard library: Operating system interfaces for path handling
import numpy as np  # numpy 1.26.3: Numerical operations for financial data manipulation
import pandas as pd  # pandas 2.1.4: Data handling and time series manipulation
from PyQt6.QtWidgets import QApplication, QMessageBox, QFileDialog  # PyQt6 6.6.1: GUI elements
import logging  # Python standard library: Logging functionality for application events
import asyncio  # Python standard library: Asynchronous I/O, event loop, and coroutines

from src.web.mfe.ui.main_window import MainWindow, run_application  # Main application window for the MFE Toolbox
from src.web.mfe.ui.armax_viewer import ARMAXViewer  # Dialog for displaying ARMAX model results
from src.web.mfe.ui.models.arma_view import ARMAView  # ARMA model configuration and visualization component

# Configure logger
logger = logging.getLogger(__name__)

# Define the path to the sample data file
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../backend/tests/test_data/market_benchmark.npy')


def configure_logging():
    """Configures the logging system for the example application"""
    # Configure logging format with timestamp and level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Set logging level to INFO
    logger.setLevel(logging.INFO)
    # Add console handler for output display
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    # Log application startup message
    logger.info("Configured logging for basic GUI usage example")


def load_sample_data():
    """Loads sample financial time series data for demonstration purposes"""
    try:
        # Check if sample data file exists at SAMPLE_DATA_PATH
        if not os.path.exists(SAMPLE_DATA_PATH):
            raise FileNotFoundError(f"Sample data file not found at: {SAMPLE_DATA_PATH}")
        # Load the NumPy array from the file using np.load
        data = np.load(SAMPLE_DATA_PATH)
        # Convert NumPy array to Pandas DataFrame with date index
        date_index = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
        df = pd.DataFrame(data, index=date_index, columns=['Sample Data'])
        # Return the DataFrame
        logger.info("Sample data loaded successfully")
        return df
    except FileNotFoundError as e:
        logger.error(f"Sample data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        raise


def show_sample_usage(window):
    """Demonstrates basic usage of the MFE Toolbox components programmatically"""
    try:
        # Load sample data using load_sample_data function
        sample_data = load_sample_data()
        # Get the ARMA view component from the window's model_views
        arma_view = window.model_views['arma']
        # Set the sample data in the ARMA view
        arma_view.set_data(sample_data)
        # Get model estimation parameters from the view
        model_params = arma_view.get_settings()
        # Trigger model estimation using handle_estimation method
        window.handle_estimation('ARMA', model_params)
        logger.info("Demonstrated ARMA model estimation")
    except Exception as e:
        logger.error(f"Error in sample usage demonstration: {e}")
        QMessageBox.critical(window, "Error", f"Error in sample usage demonstration: {e}")


def display_results_example(results):
    """Shows how to display model results programmatically"""
    try:
        # Create an ARMAXViewer instance with the results
        viewer = ARMAXViewer(results)
        # Show the viewer dialog
        viewer.show()
        logger.info("Demonstrated results viewing")
    except Exception as e:
        logger.error(f"Error displaying results: {e}")


def main():
    """Main entry point for the basic GUI usage example"""
    # Configure logging system
    configure_logging()
    # Create application arguments list
    app_args = sys.argv
    # Call run_application function with arguments
    exit_code = run_application(app_args)
    # Return the application exit code
    return exit_code


class BasicGUIExample:
    """Example class demonstrating how to use the MFE Toolbox GUI programmatically"""

    def __init__(self):
        """Initialize the example class with the main window"""
        # Create a MainWindow instance
        self.window = MainWindow()
        # Initialize data to None
        self.data = None
        # Initialize model_results to empty dictionary
        self.model_results = {}
        # Connect to MainWindow signals if needed
        # Log successful initialization
        logger.info("BasicGUIExample initialized")

    def run_demo(self):
        """Run a full demonstration of the MFE Toolbox GUI functionality"""
        try:
            # Load sample data with load_sample_data function
            self.data = load_sample_data()
            # Store data in self.data
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            return

        # Show the main window with window.show()
        self.window.show()
        # Demonstrate ARMA model estimation
        self.demonstrate_arma_estimation()
        # Demonstrate results viewing
        self.demonstrate_results_viewing()
        # Show the about dialog example
        self.demonstrate_about_dialog()
        # Log each demonstration step
        logger.info("BasicGUIExample demo completed")

    def demonstrate_arma_estimation(self):
        """Demonstrates ARMA model estimation using the GUI"""
        try:
            # Access ARMA view from window.model_views['arma']
            arma_view = self.window.model_views['arma']
            # Set data to the view using set_data method
            arma_view.set_data(self.data)
            # Configure ARMA parameters (order, constant)
            arma_order = (1, 0, 1)
            include_constant = True
            # Trigger estimation via window.handle_estimation
            self.window.handle_estimation('ARMA', arma_order, include_constant)
            # Store results in self.model_results
            self.model_results = self.window.model_results
            logger.info("Demonstrated ARMA model estimation")
        except Exception as e:
            logger.error(f"Error demonstrating ARMA estimation: {e}")

    def demonstrate_results_viewing(self):
        """Demonstrates how to view model results"""
        try:
            # Check if model_results is available
            if not self.model_results:
                raise ValueError("Model results not available. Estimate model first.")
            # Create ARMAXViewer with model_results
            viewer = ARMAXViewer(self.model_results)
            # Show the viewer dialog
            viewer.show()
            logger.info("Demonstrated results viewing")
        except Exception as e:
            logger.error(f"Error demonstrating results viewing: {e}")

    def demonstrate_about_dialog(self):
        """Demonstrates showing the about dialog"""
        try:
            # Call window.show_about_dialog()
            self.window.show_about_dialog()
            logger.info("Demonstrated about dialog")
        except Exception as e:
            logger.error(f"Error demonstrating about dialog: {e}")

    def cleanup(self):
        """Performs cleanup operations before exit"""
        # Close any open dialogs
        # Release any allocated resources
        # Log cleanup operations
        logger.info("BasicGUIExample cleanup completed")


if __name__ == '__main__':
    # Configure logging for the example
    configure_logging()
    # Create a QApplication instance
    app = QApplication(sys.argv)
    # Create an instance of the BasicGUIExample class
    example = BasicGUIExample()
    # Run the demo
    example.run_demo()
    # Exit the application
    sys.exit(app.exec())