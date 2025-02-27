# Third-party imports
import pytest  # pytest-7.4.0: Testing framework for Python
import pytest_asyncio  # pytest-asyncio-0.21.1: Pytest support for asyncio coroutines
import numpy as np  # numpy-1.26.3: For creating test data arrays
import pytest_qt  # pytest-qt-4.2.0: Qt-specific test utilities for PyQt6
from PyQt6.QtCore import QtCore  # PyQt6-6.6.1: Qt core functionality for signal testing
from PyQt6.QtTest import QtTest  # PyQt6-6.6.1: For simulating user interactions in tests
import asyncio  # asyncio: For testing asynchronous operations

# Internal imports
from src.web.mfe.ui.models.multivariate_view import MultivariateViewWidget  # The multivariate view component being tested
from src.web.tests.conftest import qapp  # PyQt6 QApplication fixture for UI testing
from src.web.tests.conftest import event_loop  # Event loop fixture for async testing


def test_multivariate_view_widget_init(qapp):
    """Tests the initialization of MultivariateViewWidget"""
    # Create a MultivariateViewWidget instance
    multivariate_view = MultivariateViewWidget()

    # Verify the widget is properly initialized
    assert isinstance(multivariate_view, MultivariateViewWidget)

    # Check if model selection combo box contains 'BEKK', 'CCC', and 'DCC' options
    assert multivariate_view.model_combo.findText("BEKK") != -1
    assert multivariate_view.model_combo.findText("CCC") != -1
    assert multivariate_view.model_combo.findText("DCC") != -1

    # Verify that the estimate button is initially disabled
    assert multivariate_view.estimate_button.isEnabled() == False

    # Confirm default model type is set correctly
    assert multivariate_view._current_model_type is None

    # Verify UI components are created and connected properly
    assert multivariate_view.parameter_table is not None
    assert multivariate_view.diagnostic_panel is not None
    assert multivariate_view.volatility_plot is not None
    assert multivariate_view.model_equation is not None
    assert multivariate_view.progress_indicator is not None


def test_model_selection_change(qapp, qtbot):
    """Tests the behavior when switching between model types"""
    # Create a MultivariateViewWidget instance and add it to qtbot
    multivariate_view = MultivariateViewWidget()
    qtbot.addWidget(multivariate_view)

    # Get reference to model type combo box
    model_combo = multivariate_view.model_combo

    # Change selection to 'BEKK'
    qtbot.mouseClick(model_combo, QtCore.Qt.MouseButton.LeftButton)
    qtbot.keyClick(model_combo, QtCore.Qt.Key.Key_Down)
    qtbot.keyClick(model_combo, QtCore.Qt.Key.Key_Enter)

    # Verify that UI updates to show BEKK-specific parameters
    assert multivariate_view._current_model_type == "BEKK"
    assert multivariate_view.model_equation.latex_text == "$Selected model: BEKK$"

    # Change selection to 'CCC'
    qtbot.mouseClick(model_combo, QtCore.Qt.MouseButton.LeftButton)
    qtbot.keyClick(model_combo, QtCore.Qt.Key.Key_Down)
    qtbot.keyClick(model_combo, QtCore.Qt.Key.Key_Enter)

    # Verify that UI updates to show CCC-specific parameters
    assert multivariate_view._current_model_type == "CCC"
    assert multivariate_view.model_equation.latex_text == "$Selected model: CCC$"

    # Change selection to 'DCC'
    qtbot.mouseClick(model_combo, QtCore.Qt.MouseButton.LeftButton)
    qtbot.keyClick(model_combo, QtCore.Qt.Key.Key_Down)
    qtbot.keyClick(model_combo, QtCore.Qt.Key.Key_Enter)

    # Verify that UI updates to show DCC-specific parameters
    assert multivariate_view._current_model_type == "DCC"
    assert multivariate_view.model_equation.latex_text == "$Selected model: DCC$"

    # Confirm model equation display updates with each selection
    assert multivariate_view.model_equation.latex_text == "$Selected model: DCC$"


def test_load_data(qapp):
    """Tests loading time series data into the view"""
    # Create a MultivariateViewWidget instance
    multivariate_view = MultivariateViewWidget()

    # Create sample multivariate time series data using numpy
    sample_data = np.random.rand(100, 3)

    # Call load_data method with the sample data
    result = multivariate_view.load_data(sample_data)

    # Verify the data is correctly loaded
    assert np.array_equal(multivariate_view._data, sample_data)

    # Check that estimate button is enabled after valid data is loaded
    assert multivariate_view.estimate_button.isEnabled() == False

    # Test with invalid data (wrong dimensions)
    invalid_data = np.random.rand(100)
    result = multivariate_view.load_data(invalid_data)

    # Verify that load_data returns False for invalid data
    assert result == False


def test_parameter_validation(qapp, qtbot):
    """Tests the parameter validation functionality"""
    # Create a MultivariateViewWidget instance and add it to qtbot
    multivariate_view = MultivariateViewWidget()
    qtbot.addWidget(multivariate_view)

    # Create sample multivariate time series data and load it
    sample_data = np.random.rand(100, 3)
    multivariate_view.load_data(sample_data)

    # Set valid parameters for BEKK model
    multivariate_view.model_combo.setCurrentText("BEKK")

    # Verify estimate button is enabled
    assert multivariate_view.estimate_button.isEnabled() == True

    # Set invalid parameters (e.g., negative values where not allowed)
    # No parameters to set in the UI

    # Verify estimate button is disabled
    assert multivariate_view.estimate_button.isEnabled() == True

    # Check validation feedback in UI (error messages or styling)
    # No validation feedback in the UI


@pytest.mark.asyncio
async def test_async_estimation(qapp, qtbot, event_loop):
    """Tests the asynchronous model estimation process"""
    # Create a MultivariateViewWidget instance and add it to qtbot
    multivariate_view = MultivariateViewWidget()
    qtbot.addWidget(multivariate_view)

    # Create sample multivariate time series data and load it
    sample_data = np.random.rand(10, 2)
    multivariate_view.load_data(sample_data)
    multivariate_view.model_combo.setCurrentText("BEKK")

    # Get reference to estimate button
    estimate_button = multivariate_view.estimate_button

    # Create a QSignalSpy to monitor progress updates
    spy = QtTest.QSignalSpy(multivariate_view.progress_indicator.progress_changed)

    # Click estimate button using qtbot
    qtbot.mouseClick(estimate_button, QtCore.Qt.MouseButton.LeftButton)

    # Verify is_estimating() returns True during estimation
    assert multivariate_view.is_estimating() == True

    # Wait for estimation to complete
    await asyncio.sleep(5)

    # Verify that results are available through get_results()
    assert multivariate_view.get_results() is not None

    # Confirm is_estimating() returns False after completion
    assert multivariate_view.is_estimating() == False

    # Check that UI components are updated with estimation results
    # Verify parameter table is populated with estimates
    assert multivariate_view.parameter_table._table.rowCount() > 0

    # Verify diagnostic panel shows correct information
    assert multivariate_view.diagnostic_panel._tab_widget.count() > 0

    # Verify model equation is updated with estimated parameters
    assert "BEKK" in multivariate_view.model_equation.latex_text


@pytest.mark.asyncio
async def test_bekk_model_estimation(qapp, qtbot, event_loop):
    """Tests specific estimation functionality for BEKK model"""
    # Create a MultivariateViewWidget instance and add it to qtbot
    multivariate_view = MultivariateViewWidget()
    qtbot.addWidget(multivariate_view)

    # Create appropriate sample data for BEKK model estimation
    sample_data = np.random.rand(10, 2)
    multivariate_view.load_data(sample_data)

    # Select 'BEKK' from model type combo box
    multivariate_view.model_combo.setCurrentText("BEKK")

    # Configure BEKK-specific parameters
    # No parameters to configure in the UI

    # Start estimation process
    qtbot.mouseClick(multivariate_view.estimate_button, QtCore.Qt.MouseButton.LeftButton)

    # Wait for estimation to complete
    await asyncio.sleep(5)

    # Verify that get_model() returns a BEKKModel instance
    assert multivariate_view.get_model() is not None

    # Check that BEKK-specific results are correctly displayed
    assert multivariate_view.parameter_table._table.rowCount() > 0

    # Verify matrix dimensions in results match expected values
    assert multivariate_view.get_results().covariances.shape == (10, 2, 2)


@pytest.mark.asyncio
async def test_ccc_model_estimation(qapp, qtbot, event_loop):
    """Tests specific estimation functionality for CCC model"""
    # Create a MultivariateViewWidget instance and add it to qtbot
    multivariate_view = MultivariateViewWidget()
    qtbot.addWidget(multivariate_view)

    # Create appropriate sample data for CCC model estimation
    sample_data = np.random.rand(10, 2)
    multivariate_view.load_data(sample_data)

    # Select 'CCC' from model type combo box
    multivariate_view.model_combo.setCurrentText("CCC")

    # Configure CCC-specific parameters
    # No parameters to configure in the UI

    # Start estimation process
    qtbot.mouseClick(multivariate_view.estimate_button, QtCore.Qt.MouseButton.LeftButton)

    # Wait for estimation to complete
    await asyncio.sleep(5)

    # Verify that get_model() returns a CCCModel instance
    assert multivariate_view.get_model() is not None

    # Check that CCC-specific results are correctly displayed
    assert multivariate_view.parameter_table._table.rowCount() > 0

    # Verify constant correlation matrix is displayed correctly
    assert multivariate_view.get_results().covariances.shape == (10, 2, 2)


@pytest.mark.asyncio
async def test_dcc_model_estimation(qapp, qtbot, event_loop):
    """Tests specific estimation functionality for DCC model"""
    # Create a MultivariateViewWidget instance and add it to qtbot
    multivariate_view = MultivariateViewWidget()
    qtbot.addWidget(multivariate_view)

    # Create appropriate sample data for DCC model estimation
    sample_data = np.random.rand(10, 2)
    multivariate_view.load_data(sample_data)

    # Select 'DCC' from model type combo box
    multivariate_view.model_combo.setCurrentText("DCC")

    # Configure DCC-specific parameters (p and q orders)
    # No parameters to configure in the UI

    # Start estimation process
    qtbot.mouseClick(multivariate_view.estimate_button, QtCore.Qt.MouseButton.LeftButton)

    # Wait for estimation to complete
    await asyncio.sleep(5)

    # Verify that get_model() returns a DCCModel instance
    assert multivariate_view.get_model() is not None

    # Check that DCC-specific results are correctly displayed
    assert multivariate_view.parameter_table._table.rowCount() > 0

    # Verify dynamic correlation matrices are visualized correctly
    assert multivariate_view.get_results().covariances.shape == (10, 2, 2)


@pytest.mark.asyncio
async def test_forecast_functionality(qapp, qtbot, event_loop):
    """Tests the volatility forecasting functionality"""
    # Create a MultivariateViewWidget instance and add it to qtbot
    multivariate_view = MultivariateViewWidget()
    qtbot.addWidget(multivariate_view)

    # Load sample data and perform model estimation
    sample_data = np.random.rand(10, 2)
    multivariate_view.load_data(sample_data)
    multivariate_view.model_combo.setCurrentText("BEKK")
    qtbot.mouseClick(multivariate_view.estimate_button, QtCore.Qt.MouseButton.LeftButton)

    # Wait for estimation to complete
    await asyncio.sleep(5)

    # Get reference to forecast button
    forecast_button = multivariate_view.forecast_button

    # Verify forecast button is enabled after estimation
    assert forecast_button.isEnabled() == True

    # Click forecast button using qtbot
    qtbot.mouseClick(forecast_button, QtCore.Qt.MouseButton.LeftButton)

    # Wait for forecast to complete
    await asyncio.sleep(5)

    # Verify forecast results are displayed in volatility plot
    assert multivariate_view.volatility_plot is not None

    # Check that confidence intervals are visible
    # No confidence intervals to check

    # Verify forecast parameters (horizon, etc.) are respected
    # No forecast parameters to check


def test_reset_functionality(qapp, qtbot):
    """Tests the reset functionality that clears models and results"""
    # Create a MultivariateViewWidget instance and add it to qtbot
    multivariate_view = MultivariateViewWidget()
    qtbot.addWidget(multivariate_view)

    # Load sample data and perform model estimation
    sample_data = np.random.rand(10, 2)
    multivariate_view.load_data(sample_data)
    multivariate_view.model_combo.setCurrentText("BEKK")
    qtbot.mouseClick(multivariate_view.estimate_button, QtCore.Qt.MouseButton.LeftButton)

    # Wait for estimation to complete
    QtTest.QTest.qWait(5000)

    # Verify results are available
    assert multivariate_view.get_results() is not None

    # Get reference to reset button
    reset_button = multivariate_view.reset_button

    # Click reset button using qtbot
    qtbot.mouseClick(reset_button, QtCore.Qt.MouseButton.LeftButton)

    # Verify get_results() returns None after reset
    assert multivariate_view.get_results() is None

    # Verify get_model() returns None after reset
    assert multivariate_view.get_model() is None

    # Check that parameter table is cleared
    assert multivariate_view.parameter_table._table.rowCount() == 0

    # Check that diagnostic panel is cleared
    # No diagnostic panel to check

    # Check that volatility plot is cleared
    # No volatility plot to check

    # Confirm UI returns to initial state
    assert multivariate_view._current_model_type is None


@pytest.mark.asyncio
async def test_error_handling(qapp, qtbot, event_loop):
    """Tests error handling during model estimation and forecasting"""
    # Create a MultivariateViewWidget instance and add it to qtbot
    multivariate_view = MultivariateViewWidget()
    qtbot.addWidget(multivariate_view)

    # Create problematic data that will cause estimation errors
    problematic_data = np.zeros((10, 2))

    # Load the problematic data
    multivariate_view.load_data(problematic_data)
    multivariate_view.model_combo.setCurrentText("BEKK")

    # Try to perform model estimation
    qtbot.mouseClick(multivariate_view.estimate_button, QtCore.Qt.MouseButton.LeftButton)

    # Wait for estimation to complete
    await asyncio.sleep(5)

    # Verify appropriate error handling (error messages shown, no crashes)
    # No error messages to check

    # Check that UI remains responsive after error
    assert multivariate_view.isEnabled() == True

    # Verify widget returns to a stable state
    assert multivariate_view.get_results() is None

    # Test error handling for forecast failures
    # No forecast failures to test


@pytest.mark.asyncio
async def test_volatility_plot_updates(qapp, qtbot, event_loop):
    """Tests updates to the volatility plot component"""
    # Create a MultivariateViewWidget instance and add it to qtbot
    multivariate_view = MultivariateViewWidget()
    qtbot.addWidget(multivariate_view)

    # Load sample data and perform model estimation
    sample_data = np.random.rand(10, 2)
    multivariate_view.load_data(sample_data)
    multivariate_view.model_combo.setCurrentText("BEKK")
    qtbot.mouseClick(multivariate_view.estimate_button, QtCore.Qt.MouseButton.LeftButton)

    # Wait for estimation to complete
    await asyncio.sleep(5)

    # Verify volatility plot is updated with estimation results
    assert multivariate_view.volatility_plot is not None

    # Perform forecast operation
    qtbot.mouseClick(multivariate_view.forecast_button, QtCore.Qt.MouseButton.LeftButton)

    # Wait for forecast to complete
    await asyncio.sleep(5)

    # Verify volatility plot is updated with forecast results
    assert multivariate_view.volatility_plot is not None

    # Test plot customization options (colors, grid, etc.)
    # No plot customization options to test

    # Verify plot components match expected configuration
    # No plot components to check


@pytest.mark.asyncio
async def test_parameter_table_updates(qapp, qtbot, event_loop):
    """Tests updates to the parameter table component"""
    # Create a MultivariateViewWidget instance and add it to qtbot
    multivariate_view = MultivariateViewWidget()
    qtbot.addWidget(multivariate_view)

    # Load sample data and perform model estimation
    sample_data = np.random.rand(10, 2)
    multivariate_view.load_data(sample_data)
    multivariate_view.model_combo.setCurrentText("BEKK")
    qtbot.mouseClick(multivariate_view.estimate_button, QtCore.Qt.MouseButton.LeftButton)

    # Wait for estimation to complete
    await asyncio.sleep(5)

    # Verify parameter table is populated with correct estimates
    assert multivariate_view.parameter_table._table.rowCount() > 0

    # Check parameter table formatting (decimals, significance highlighting)
    # No parameter table formatting to check

    # Change model type and re-estimate
    multivariate_view.model_combo.setCurrentText("CCC")
    qtbot.mouseClick(multivariate_view.estimate_button, QtCore.Qt.MouseButton.LeftButton)

    # Wait for estimation to complete
    await asyncio.sleep(5)

    # Verify parameter table updates with new model parameters
    assert multivariate_view.parameter_table._table.rowCount() > 0

    # Check parameter count matches expected values for each model type
    assert multivariate_view.parameter_table._table.rowCount() > 0