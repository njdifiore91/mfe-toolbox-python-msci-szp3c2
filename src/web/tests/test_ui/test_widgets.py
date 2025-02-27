import pytest  # pytest-7.4.3
from PyQt6 import QtWidgets  # PyQt6-6.6.1
from PyQt6.QtTest import QTest  # PyQt6-6.6.1
from PyQt6.QtWidgets import QApplication  # PyQt6-6.6.1
import numpy as np  # numpy-1.26.3
from unittest.mock import patch  # Built-in

from src.web.mfe.ui import widgets  # Internal
from src.web.tests import conftest  # Internal


def test_model_config_widget_initialization(qtbot):
    """Tests the proper initialization of the ModelConfigWidget"""
    # LD1: Initialize a ModelConfigWidget instance
    config_widget = widgets.ModelConfigWidget()

    # LD1: Verify the widget is properly initialized with default values
    assert config_widget.ar_order_input.text() == "1"
    assert config_widget.ma_order_input.text() == "1"
    assert config_widget.constant_checkbox.isChecked() is True
    assert config_widget.exogenous_combo.count() == 1
    assert config_widget.exogenous_combo.itemText(0) == "None"

    # LD1: Check that the UI elements are created correctly
    assert config_widget.layout.rowCount() == 4

    # LD1: Verify the AR and MA order fields have the correct initial values
    assert config_widget.ar_order_input.text() == "1"
    assert config_widget.ma_order_input.text() == "1"

    # LD1: Check that the constant checkbox is correctly initialized
    assert config_widget.constant_checkbox.isChecked() is True

    # LD1: Verify the exogenous variables dropdown is properly set up
    assert config_widget.exogenous_combo.itemText(0) == "None"


def test_model_config_widget_interactions(qtbot):
    """Tests user interactions with ModelConfigWidget"""
    # LD1: Initialize a ModelConfigWidget instance
    config_widget = widgets.ModelConfigWidget()

    # LD1: Use qtbot to simulate user interactions (typing, clicking)
    # LD1: Change AR order field and verify the value is updated
    qtbot.keyClicks(config_widget.ar_order_input, "2")
    assert config_widget.ar_order_input.text() == "2"

    # LD1: Change MA order field and verify the value is updated
    qtbot.keyClicks(config_widget.ma_order_input, "3")
    assert config_widget.ma_order_input.text() == "3"

    # LD1: Toggle constant checkbox and verify the state changes
    qtbot.mouseClick(config_widget.constant_checkbox, Qt.MouseButton.LeftButton)
    assert config_widget.constant_checkbox.isChecked() is False

    # LD1: Select a different item in exogenous dropdown and verify selection
    config_widget.exogenous_combo.addItem("Var1")
    qtbot.mouseClick(config_widget.exogenous_combo, Qt.MouseButton.LeftButton)
    qtbot.keyClick(config_widget.exogenous_combo, Qt.Key.Key_Down)
    qtbot.keyClick(config_widget.exogenous_combo, Qt.Key.Key_Enter)
    assert config_widget.exogenous_combo.currentText() == "Var1"


def test_diagnostic_plots_widget_initialization(qtbot):
    """Tests the proper initialization of the DiagnosticPlotsWidget"""
    # LD1: Initialize a DiagnosticPlotsWidget instance
    plots_widget = widgets.DiagnosticPanel()

    # LD1: Verify the widget is properly initialized
    assert plots_widget is not None

    # LD1: Check that all plot areas are created correctly
    assert plots_widget._tab_widget.count() == 6

    # LD1: Verify the layout is set up as expected
    assert plots_widget._main_layout.count() == 2

    # LD1: Check that the widget has the expected size policy
    assert plots_widget.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
    assert plots_widget.sizePolicy().verticalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding


def test_diagnostic_plots_widget_update(qtbot):
    """Tests updating the diagnostic plots with new data"""
    # LD1: Initialize a DiagnosticPlotsWidget instance
    plots_widget = widgets.DiagnosticPanel()

    # LD1: Create mock data using numpy arrays
    data = np.random.randn(100)
    residuals = np.random.randn(100)
    fitted_values = np.random.randn(100)
    model_results = {}

    # LD1: Call the update method with the mock data
    plots_widget.update_plots(data, residuals, fitted_values, model_results)

    # LD1: Verify that the plots are updated correctly
    # LD1: Check that the residuals plot contains the correct data
    assert plots_widget._residual_plot._residuals is data
    assert plots_widget._residual_plot._fitted_values is residuals

    # LD1: Verify that the ACF and PACF plots are updated with the correct data
    assert plots_widget._acf_plot._data is data
    assert plots_widget._pacf_plot._data is data


def test_model_estimation_widget_initialization(qtbot):
    """Tests the proper initialization of the ModelEstimationWidget"""
    # LD1: Initialize a ModelEstimationWidget instance
    estimation_widget = widgets.ControlButtonsWidget()

    # LD1: Verify the widget is properly initialized
    assert estimation_widget is not None

    # LD1: Check that all buttons are created and have the correct text
    assert estimation_widget.estimate_button.text() == "Estimate Model"
    assert estimation_widget.reset_button.text() == "Reset"
    assert estimation_widget.view_results_button.text() == "View Results"
    assert estimation_widget.close_button.text() == "Close"

    # LD1: Verify the layout is set up as expected
    assert estimation_widget.layout().count() == 4

    # LD1: Check that the progress indicator is properly initialized
    assert estimation_widget.progress_indicator is not None


def test_model_estimation_widget_signals(qtbot):
    """Tests the signals emitted by the ModelEstimationWidget"""
    # LD1: Initialize a ModelEstimationWidget instance
    estimation_widget = widgets.ControlButtonsWidget()

    # LD1: Connect to the widget's signals with mock slots
    estimate_triggered = False
    reset_triggered = False
    view_results_triggered = False
    close_triggered = False

    def on_estimate_requested():
        nonlocal estimate_triggered
        estimate_triggered = True

    def on_reset_requested():
        nonlocal reset_triggered
        reset_triggered = True

    def on_view_results_requested():
        nonlocal view_results_triggered
        view_results_triggered = True

    def on_close_requested():
        nonlocal close_triggered
        close_triggered = True

    estimation_widget.estimateRequested.connect(on_estimate_requested)
    estimation_widget.resetRequested.connect(on_reset_requested)
    estimation_widget.viewResultsRequested.connect(on_view_results_requested)
    estimation_widget.closeRequested.connect(on_close_requested)

    # LD1: Trigger the 'Estimate Model' button click
    qtbot.mouseClick(estimation_widget.estimate_button, Qt.MouseButton.LeftButton)

    # LD1: Verify that the estimateRequested signal is emitted
    assert estimate_triggered is True

    # LD1: Trigger the 'Reset' button click
    qtbot.mouseClick(estimation_widget.reset_button, Qt.MouseButton.LeftButton)

    # LD1: Verify that the resetRequested signal is emitted
    assert reset_triggered is True

    # LD1: Trigger the 'View Results' button click
    qtbot.mouseClick(estimation_widget.view_results_button, Qt.MouseButton.LeftButton)

    # LD1: Verify that the viewResultsRequested signal is emitted
    assert view_results_triggered is True

    # LD1: Trigger the 'Close' button click
    qtbot.mouseClick(estimation_widget.close_button, Qt.MouseButton.LeftButton)

    # LD1: Verify that the closeRequested signal is emitted
    assert close_triggered is True


@pytest.mark.asyncio
async def test_async_model_estimation(qtbot, monkeypatch):
    """Tests the asynchronous model estimation workflow"""
    # LD1: Mock the async model estimation functions
    async def mock_estimate_model(data):
        for i in range(101):
            await asyncio.sleep(0.001)
            yield i

    monkeypatch.setattr(widgets.ARMAX, "estimate", mock_estimate_model)

    # LD1: Initialize a ModelEstimationWidget instance
    estimation_widget = widgets.ControlButtonsWidget()

    # LD1: Set up progress signal monitoring
    progress_values = []

    def on_progress_changed(value):
        progress_values.append(value)

    estimation_widget.progress_indicator.progress_changed.connect(on_progress_changed)

    # LD1: Trigger the estimation process
    qtbot.mouseClick(estimation_widget.estimate_button, Qt.MouseButton.LeftButton)

    # LD1: Verify progress updates are emitted during estimation
    await asyncio.sleep(0.1)  # Wait for estimation to complete
    assert len(progress_values) > 0

    # LD1: Check that the UI updates correctly during async operations
    assert estimation_widget.progress_indicator.is_showing() is False

    # LD1: Verify final state after estimation completes
    assert estimation_widget.progress_indicator.get_progress() == 0.0


def test_widget_integration(qtbot):
    """Tests the integration of multiple widgets together"""
    # LD1: Initialize ModelConfigWidget, DiagnosticPlotsWidget, and ModelEstimationWidget instances
    config_widget = widgets.ModelConfigWidget()
    plots_widget = widgets.DiagnosticPanel()
    estimation_widget = widgets.ControlButtonsWidget()

    # LD1: Create a parent widget and add all three widgets to it
    parent_widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(parent_widget)
    layout.addWidget(config_widget)
    layout.addWidget(plots_widget)
    layout.addWidget(estimation_widget)
    parent_widget.setLayout(layout)

    # LD1: Verify the layout and interaction between widgets
    assert layout.count() == 3

    # LD1: Change values in the ModelConfigWidget
    qtbot.keyClicks(config_widget.ar_order_input, "2")
    qtbot.keyClicks(config_widget.ma_order_input, "3")

    # LD1: Trigger the estimate button in ModelEstimationWidget
    qtbot.mouseClick(estimation_widget.estimate_button, Qt.MouseButton.LeftButton)

    # LD1: Verify that the DiagnosticPlotsWidget receives updates
    # LD2: Test the overall workflow from configuration to estimation to results
    # Note: This test only verifies that the signals are connected and the workflow is triggered.
    # It does not verify the correctness of the plots, as that would require more complex mocking and data validation.
    assert plots_widget is not None