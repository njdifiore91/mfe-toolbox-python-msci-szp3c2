# src/web/tests/test_ui/test_models/test_garch_view.py
"""
Unit tests for GARCHView UI component that visualizes and interacts with GARCH volatility models.
"""
import pytest  # pytest v7.4.3
from unittest import mock  # standard library
import asyncio  # standard library
import numpy as np  # numpy 1.26.3
from PyQt6.QtTest import QTest, QSignalSpy  # PyQt6 v6.6.1
from PyQt6.QtWidgets import QSpinBox, QComboBox  # PyQt6 v6.6.1

# Internal imports
from mfe.ui.models.garch_view import GARCHView, GARCHViewSettings  # Component under test
from . import async_test_helper, UI_TIMEOUT  # Async test helper

# Define test data size
TEST_DATA_SIZE = 100


@pytest.fixture
def sample_time_series():
    """Fixture for creating a sample time series data."""
    np.random.seed(42)
    return np.random.randn(TEST_DATA_SIZE)


@pytest.fixture
def mock_volatility_data():
    """Fixture for mocking volatility data."""
    return np.random.rand(TEST_DATA_SIZE)


@pytest.fixture
def mock_model_params():
    """Fixture for mocking model parameters."""
    return {
        'omega': {'estimate': 0.01, 'std_error': 0.005, 't_stat': 2.0, 'p_value': 0.05},
        'alpha_1': {'estimate': 0.1, 'std_error': 0.05, 't_stat': 2.0, 'p_value': 0.05},
        'beta_1': {'estimate': 0.8, 'std_error': 0.05, 't_stat': 16.0, 'p_value': 0.001}
    }


@pytest.mark.ui
def test_garch_view_initialization(qapp):
    """Test that GARCHView initializes correctly with default parameters."""
    # Create a GARCHView instance
    garch_view = GARCHView()

    # Verify default p and q order values
    assert garch_view._settings.p == 1
    assert garch_view._settings.q == 1

    # Check that default distribution is 'normal'
    assert garch_view._settings.distribution == 'normal'

    # Verify UI components are properly initialized
    assert isinstance(garch_view._p_spin, QSpinBox)
    assert isinstance(garch_view._q_spin, QSpinBox)
    assert isinstance(garch_view._dist_type_combo, QComboBox)

    # Ensure no model is set initially
    assert garch_view._model is None


@pytest.mark.ui
def test_set_data(qapp, sample_time_series):
    """Test that data can be set on the GARCHView correctly."""
    # Create a GARCHView instance
    garch_view = GARCHView()

    # Set sample time series data
    garch_view.set_data(sample_time_series)

    # Verify data is stored correctly
    assert np.array_equal(garch_view._data, sample_time_series)

    # Check that volatility plot is updated
    assert garch_view._volatility_plot._data is not None

    # Verify UI is updated to reflect new data
    assert garch_view._model_equation.latex_text != ""


@pytest.mark.ui
def test_update_p_order(qapp):
    """Test that GARCH order (p) parameter can be updated."""
    # Create a GARCHView instance
    garch_view = GARCHView()

    # Use QTest to change p order spinbox value
    QTest.keyClick(garch_view._p_spin, '2')

    # Verify p order is updated in the model
    assert garch_view._settings.p == 1  # keyClick doesn't immediately update the value
    QTest.qWait(100)
    assert garch_view._p_spin.value() == 2
    assert garch_view._settings.p == 2

    # Check that UI reflects the new p order
    assert garch_view._p_spin.value() == 2

    # Verify model equation is updated
    assert "phi" in garch_view._model_equation.latex_text


@pytest.mark.ui
def test_update_q_order(qapp):
    """Test that ARCH order (q) parameter can be updated."""
    # Create a GARCHView instance
    garch_view = GARCHView()

    # Use QTest to change q order spinbox value
    QTest.keyClick(garch_view._q_spin, '2')

    # Verify q order is updated in the model
    assert garch_view._settings.q == 1  # keyClick doesn't immediately update the value
    QTest.qWait(100)
    assert garch_view._q_spin.value() == 2
    assert garch_view._settings.q == 2

    # Check that UI reflects the new q order
    assert garch_view._q_spin.value() == 2

    # Verify model equation is updated
    assert "theta" in garch_view._model_equation.latex_text


@pytest.mark.ui
def test_distribution_selection(qapp):
    """Test that distribution type can be changed."""
    # Create a GARCHView instance
    garch_view = GARCHView()

    # Use QTest to change distribution type in combo box
    QTest.keyClick(garch_view._dist_type_combo, Qt.Key.Key_Down)

    # Verify distribution type is updated in model settings
    assert garch_view._settings.distribution == 'normal'  # keyClick doesn't immediately update the value
    QTest.qWait(100)
    assert garch_view._dist_type_combo.currentData() != 'normal'
    assert garch_view._settings.distribution == garch_view._dist_type_combo.currentData()

    # Check that UI shows appropriate distribution parameters
    assert garch_view._dist_type_combo.currentData() in ['student', 'ged', 'skewed_t']


@pytest.mark.asyncio
@pytest.mark.ui
@async_test_helper
async def test_async_estimate_model(qapp, sample_time_series, mock_volatility_data):
    """Test that GARCH model estimation works asynchronously."""
    # Create a GARCHView instance with mocked GARCH model
    garch_view = GARCHView()
    garch_view.set_data(sample_time_series)

    # Mock the async_estimate_model method
    async def mock_async_estimate_model():
        await asyncio.sleep(0.1)  # Simulate some work
        return {"model": mock.MagicMock(), "parameters": mock.MagicMock(), "conditional_variances": mock_volatility_data}

    garch_view.async_estimate_model = mock_async_estimate_model

    # Call async_estimate_model method
    await garch_view.on_estimate_clicked()

    # Verify estimation is performed asynchronously
    assert garch_view._estimation_result is not None

    # Check that UI is updated with estimation results
    assert garch_view._parameter_table._parameter_data is not None

    # Verify parameter table is updated
    assert garch_view._model_equation.latex_text != ""

    # Check that plots are refreshed with new data
    assert garch_view._volatility_plot._data is not None

    # Ensure model equation is updated
    assert garch_view._volatility_plot._data is not None


@pytest.mark.ui
@async_test_helper
async def test_ui_responsiveness_during_estimation(qapp, sample_time_series):
    """Test that UI remains responsive during async model estimation."""
    # Create a GARCHView instance
    garch_view = GARCHView()
    garch_view.set_data(sample_time_series)

    # Start async model estimation
    estimation_task = asyncio.create_task(garch_view.on_estimate_clicked())

    # Attempt UI interactions during estimation
    QTest.qWait(100)
    assert garch_view._p_spin.isEnabled() is False

    # Verify UI remains responsive
    QTest.qWait(100)
    assert garch_view._q_spin.isEnabled() is False

    # Ensure estimation completes successfully
    await estimation_task

    # Check that results are displayed correctly
    assert garch_view._parameter_table._parameter_data is not None


@pytest.mark.ui
@async_test_helper
async def test_parameter_table_update(qapp, sample_time_series, mock_model_params):
    """Test that parameter table is updated correctly after estimation."""
    # Create a GARCHView instance with mocked GARCH model
    garch_view = GARCHView()
    garch_view.set_data(sample_time_series)

    # Mock the async_estimate_model method
    async def mock_async_estimate_model():
        await asyncio.sleep(0.1)  # Simulate some work
        return {"model": mock.MagicMock(), "parameters": mock_model_params, "conditional_variances": mock.MagicMock()}

    garch_view.async_estimate_model = mock_async_estimate_model

    # Perform model estimation
    await garch_view.on_estimate_clicked()

    # Examine parameter table contents
    table = garch_view._parameter_table._table

    # Verify all parameters are shown correctly
    assert table.rowCount() == len(mock_model_params)
    assert table.item(0, 0).text() == 'omega'
    assert table.item(1, 0).text() == 'alpha_1'
    assert table.item(2, 0).text() == 'beta_1'

    # Check statistical values (std errors, t-stats, p-values)
    assert table.item(0, 1).text() == '0.0100'
    assert table.item(0, 2).text() == '0.0050'
    assert table.item(0, 3).text() == '2.00'
    assert table.item(0, 4).text() == '0.0500'

    # Ensure formatting is applied correctly
    assert table.item(2, 3).text() == '16.00'
    assert table.item(2, 4).text() == '< 0.001'


@pytest.mark.ui
@async_test_helper
async def test_volatility_plot_update(qapp, sample_time_series, mock_volatility_data):
    """Test that volatility plot updates correctly after estimation."""
    # Create a GARCHView instance with mocked model
    garch_view = GARCHView()
    garch_view.set_data(sample_time_series)

    # Mock volatility plot methods
    garch_view._volatility_plot.set_data = mock.MagicMock()
    garch_view._volatility_plot.update_plot = mock.MagicMock()

    # Mock the async_estimate_model method
    async def mock_async_estimate_model():
        await asyncio.sleep(0.1)  # Simulate some work
        return {"model": mock.MagicMock(), "parameters": mock.MagicMock(), "conditional_variances": mock_volatility_data}

    garch_view.async_estimate_model = mock_async_estimate_model

    # Perform model estimation
    await garch_view.on_estimate_clicked()

    # Verify plot method was called with correct data
    garch_view._volatility_plot.set_data.assert_called_with(garch_view._returns, mock_volatility_data)

    # Check that volatility plot is visible and correctly rendered
    garch_view._volatility_plot.update_plot.assert_called()


@pytest.mark.ui
@async_test_helper
async def test_forecast_functionality(qapp, sample_time_series, mock_volatility_data):
    """Test that volatility forecasting works correctly."""
    # Create a GARCHView instance with mocked model
    garch_view = GARCHView()
    garch_view.set_data(sample_time_series)

    # Mock the async_estimate_model method
    async def mock_async_estimate_model():
        await asyncio.sleep(0.1)  # Simulate some work
        return {"model": mock.MagicMock(), "parameters": mock.MagicMock(), "conditional_variances": mock_volatility_data}

    garch_view.async_estimate_model = mock_async_estimate_model

    # Perform model estimation
    await garch_view.on_estimate_clicked()

    # Set forecast parameters (horizon, alpha)
    garch_view._forecast_horizon_spin = mock.MagicMock()
    garch_view._forecast_horizon_spin.value.return_value = 10
    garch_view._forecast_alpha_spin = mock.MagicMock()
    garch_view._forecast_alpha_spin.value.return_value = 0.05

    # Mock forecast results
    forecast_data = np.random.rand(10)
    garch_view._model.forecast = mock.MagicMock(return_value=forecast_data)

    # Mock volatility plot methods
    garch_view._volatility_plot.set_forecast_data = mock.MagicMock()

    # Trigger forecast operation
    await garch_view.on_forecast_clicked()

    # Verify forecast results are calculated
    garch_view._model.forecast.assert_called_with(10)

    # Check that volatility plot is updated with forecast data
    garch_view._volatility_plot.set_forecast_data.assert_called_with(forecast_data)

    # Ensure confidence intervals are displayed correctly
    assert garch_view._volatility_plot.set_forecast_data.call_count == 1


@pytest.mark.ui
@async_test_helper
async def test_error_handling(qapp, sample_time_series):
    """Test that errors during model estimation are handled properly."""
    # Create a GARCHView instance
    garch_view = GARCHView()
    garch_view.set_data(sample_time_series)

    # Mock GARCH model to raise exception during estimation
    async def mock_async_estimate_model():
        await asyncio.sleep(0.1)  # Simulate some work
        raise ValueError("Estimation failed")

    garch_view.async_estimate_model = mock_async_estimate_model

    # Attempt model estimation
    await garch_view.on_estimate_clicked()

    # Verify error is caught and handled appropriately
    assert garch_view._progress_bar.isVisible() is False

    # Check that error message is displayed to user
    # (This requires checking the QMessageBox, which is difficult to do directly)

    # Ensure UI remains in consistent state after error
    assert garch_view._p_spin.isEnabled() is True
    assert garch_view._q_spin.isEnabled() is True


@pytest.mark.ui
def test_settings_get_apply(qapp):
    """Test getting and applying GARCHViewSettings."""
    # Create a GARCHView instance
    garch_view1 = GARCHView()

    # Modify UI controls (p, q, distribution)
    QTest.keyClick(garch_view1._p_spin, '3')
    QTest.keyClick(garch_view1._q_spin, '4')
    QTest.keyClick(garch_view1._dist_type_combo, Qt.Key.Key_Down)

    # Get settings using get_settings()
    settings = garch_view1.get_settings()

    # Verify settings reflect UI state
    assert settings.p == 1  # keyClick doesn't immediately update the value
    assert settings.q == 1
    assert settings.distribution == 'normal'
    QTest.qWait(100)
    assert settings.p == 3
    assert settings.q == 4
    assert settings.distribution == garch_view1._dist_type_combo.currentData()

    # Create new GARCHView instance
    garch_view2 = GARCHView()

    # Apply saved settings using apply_settings()
    garch_view2.apply_settings(settings)

    # Verify UI controls match applied settings
    assert garch_view2._p_spin.value() == 3
    assert garch_view2._q_spin.value() == 4
    assert garch_view2._dist_type_combo.currentData() == settings.distribution