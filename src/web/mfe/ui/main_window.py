"""
Main application window for the MFE Toolbox, providing the primary user interface for financial econometric analysis.
Implements a PyQt6-based QMainWindow that hosts various model configuration, estimation, and visualization components.
"""
import logging
import os
import sys
import asyncio  # 3.12.0: Asynchronous I/O, event loop, and coroutines

from PyQt6.QtWidgets import QMainWindow, QApplication, QTabWidget, QAction, QMenuBar, QToolBar, QStatusBar, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox  # 6.6.1: Qt widgets
from PyQt6.QtGui import QIcon, QKeySequence  # 6.6.1: Qt GUI components
from PyQt6.QtCore import pyqtSignal, Qt, QSettings, QSize, QRect  # 6.6.1: Qt core components

from ..about_dialog import AboutDialog  # Creates and displays the About dialog
from ..close_dialog import CloseDialog  # Creates and displays the close confirmation dialog
from ..armax_viewer import ArmaxViewer  # Displays ARMAX model results in a detailed view
from ..components.navigation_bar import NavigationBar  # Provides navigation controls for the main window
from ..components.error_display import ErrorDisplay  # Displays error messages in the UI
from ..async.task_manager import TaskManager  # Manages asynchronous tasks for long-running operations
from ..models.arma_view import ArmaView  # ARMA model configuration and estimation view
from ..models.garch_view import GarchView  # GARCH model configuration and estimation view
from ..models.multivariate_view import MultivariateView  # Multivariate model configuration and estimation view
from ..styles import STYLES  # CSS-style definitions for the UI components

# Configure logger
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window for the MFE Toolbox, providing access to all financial econometric analysis tools
    """

    def __init__(self, parent=None):
        """
        Initializes the main window and sets up the UI components
        """
        super().__init__(parent)

        # Initialize UI components
        self.task_manager = TaskManager()
        self.tab_widget = QTabWidget()
        self.model_views = {}
        self.error_display = ErrorDisplay()
        self.settings = QSettings("MFE_Toolbox", "MFE_Application")
        self.has_unsaved_changes = False

        # Set up the menu bar and toolbars
        self.setup_menu()
        self.setup_toolbar()

        # Create model views (ARMA, GARCH, Multivariate)
        self.create_model_views()

        # Set up the tab widget
        self.init_ui()

        # Connect signals and slots
        self.tab_widget.currentChanged.connect(self.handle_tab_changed)

        # Set up window properties (title, size, icon)
        self.setWindowTitle("MFE Toolbox")
        self.setMinimumSize(QSize(800, 600))
        app_icon = QIcon(os.path.join(sys._MEIPASS, 'assets/icons/app_icon.png')) if getattr(sys, 'frozen', False) else QIcon('src/web/mfe/ui/assets/icons/app_icon.png')
        self.setWindowIcon(app_icon)

        # Restore window state from settings if available
        self.restore_window_state()

        # Initialize the error display component
        self.error_display = ErrorDisplay()

        # Set has_unsaved_changes to False
        self.has_unsaved_changes = False

        logger.info("Main window initialized")

    def init_ui(self):
        """
        Sets up the user interface components
        """
        # Create the central widget
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Add the tab widget
        main_layout.addWidget(self.tab_widget)

        # Set up the status bar
        self.setStatusBar(QStatusBar(self))

        # Apply styles from the STYLES dictionary
        self.setStyleSheet(STYLES.get("main_window", ""))
        self.tab_widget.setStyleSheet(STYLES.get("tab_widget", ""))

        self.setCentralWidget(central_widget)
        logger.debug("UI initialized")

    def setup_menu(self):
        """
        Creates the menu bar and menus
        """
        menu_bar = QMenuBar(self)

        # File menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction(QIcon.fromTheme("document-open"), "Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip("Open a file")
        open_action.triggered.connect(self.open_data)
        file_menu.addAction(open_action)

        save_action = QAction(QIcon.fromTheme("document-save"), "Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.setStatusTip("Save the current analysis")
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)

        exit_action = QAction(QIcon.fromTheme("application-exit"), "Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction(QIcon.fromTheme("help-about"), "&About", self)
        about_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        about_action.setStatusTip("About MFE Toolbox")
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        self.setMenuBar(menu_bar)
        logger.debug("Menu setup complete")

    def setup_toolbar(self):
        """
        Creates the toolbar with common actions
        """
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)

        open_action = QAction(QIcon.fromTheme("document-open"), "Open", self)
        open_action.setStatusTip("Open a file")
        open_action.triggered.connect(self.open_data)
        toolbar.addAction(open_action)

        save_action = QAction(QIcon.fromTheme("document-save"), "Save", self)
        save_action.setStatusTip("Save the current analysis")
        save_action.triggered.connect(self.save_results)
        toolbar.addAction(save_action)

        estimate_action = QAction(QIcon.fromTheme("system-run"), "Estimate", self)
        estimate_action.setStatusTip("Estimate the model")
        #estimate_action.triggered.connect(self.estimate_model)
        toolbar.addAction(estimate_action)

        help_action = QAction(QIcon.fromTheme("help-contents"), "Help", self)
        help_action.setStatusTip("Help")
        help_action.triggered.connect(self.show_about_dialog)
        toolbar.addAction(help_action)

        self.addToolBar(toolbar)
        logger.debug("Toolbar setup complete")

    def create_model_views(self):
        """
        Initializes all model view components
        """
        self.arma_view = ArmaView()
        self.garch_view = GarchView()
        self.multivariate_view = MultivariateView()

        self.model_views["arma"] = self.arma_view
        self.model_views["garch"] = self.garch_view
        self.model_views["multivariate"] = self.multivariate_view

        self.tab_widget.addTab(self.arma_view, "ARMA")
        self.tab_widget.addTab(self.garch_view, "GARCH")
        self.tab_widget.addTab(self.multivariate_view, "Multivariate")

        self.arma_view.estimate_signal.connect(self.handle_estimation)
        self.garch_view.estimate_signal.connect(self.handle_estimation)
        self.multivariate_view.estimate_signal.connect(self.handle_estimation)
        logger.debug("Model views created")

    def handle_estimation(self, model_type, parameters):
        """
        Handles model estimation requests from model views
        """
        logger.info(f"Estimation requested for model type: {model_type}")
        self.statusBar().showMessage(f"Estimating {model_type} model...")

        # Create an async task for model estimation
        async def estimate_task():
            return await self.async_estimate_model(model_type, parameters)

        # Run the task using the task manager
        self.task_manager.run_task(estimate_task)

        # Connect task signals to handle_estimation_complete and handle_estimation_error
        self.task_manager.task_completed.connect(self.handle_estimation_complete)
        self.task_manager.task_error.connect(self.handle_estimation_error)

    async def async_estimate_model(self, model_type, parameters):
        """
        Asynchronous method to perform model estimation
        """
        # Select the appropriate estimation function based on model_type
        if model_type == "ARMA":
            estimation_function = self.model_views["arma"].estimate_model
        elif model_type == "GARCH":
            estimation_function = self.model_views["garch"].estimate_model
        elif model_type == "Multivariate":
            estimation_function = self.model_views["multivariate"].estimate_model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Perform parameter validation
        # (Implementation depends on the specific model)

        # Execute the estimation asynchronously
        results = await estimation_function(parameters)

        # Monitor and report progress
        # (Implementation depends on the specific model)

        return results

    def handle_estimation_complete(self, results):
        """
        Handles the completion of a model estimation task
        """
        self.statusBar().showMessage("Estimation complete")
        # Display results using ArmaxViewer
        viewer = ArmaxViewer(results)
        viewer.show()
        # Update the current model view with results
        # (Implementation depends on the specific model)
        # Set has_unsaved_changes to True
        self.has_unsaved_changes = True

    def handle_estimation_error(self, error):
        """
        Handles errors during model estimation
        """
        logger.error(f"Estimation error: {error}")
        self.statusBar().showMessage(f"Estimation failed: {error}")
        self.error_display.show_error(str(error))
        # Reset UI state to allow for new estimation
        # (Implementation depends on the specific model)

    def show_about_dialog(self):
        """
        Shows the about dialog
        """
        dialog = AboutDialog(self)
        dialog.exec()

    def open_data(self):
        """
        Opens data for analysis
        """
        # Show a file dialog to select a data file
        # Load the selected data file
        # Pass the data to the current model view
        # Update status bar
        pass

    def save_results(self):
        """
        Saves analysis results
        """
        # Show a file dialog to select save location
        # Collect results from the current model view
        # Save results to the selected file
        # Update status bar
        # Set has_unsaved_changes to False
        pass

    def handle_tab_changed(self, index):
        """
        Handles tab widget selection changes
        """
        # Get the selected model view
        # Update the status bar with the current model type
        # Update menus and toolbars based on the selected model
        pass

    def closeEvent(self, event):
        """
        Override of closeEvent to handle application exit with unsaved changes
        """
        if self.has_unsaved_changes:
            close_dialog = CloseDialog(self)
            result = close_dialog.exec()
            if result:
                self.save_window_state()
                event.accept()
            else:
                event.ignore()
        else:
            self.save_window_state()
            event.accept()

    def save_window_state(self):
        """
        Saves the window geometry and state to QSettings
        """
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("currentTabIndex", self.tab_widget.currentIndex())
        logger.debug("Window state saved")

    def restore_window_state(self):
        """
        Restores the window geometry and state from QSettings
        """
        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry", QByteArray()))
        if self.settings.contains("windowState"):
            self.restoreState(self.settings.value("windowState", QByteArray()))
        if self.settings.contains("currentTabIndex"):
            index = int(self.settings.value("currentTabIndex", 0))
            self.tab_widget.setCurrentIndex(index)
        logger.debug("Window state restored")


def run_application(args):
    """
    Entry point function to run the MFE Toolbox application
    """
    app = QApplication(args)
    main_window = MainWindow()
    main_window.show()
    exit_code = app.exec()
    return exit_code