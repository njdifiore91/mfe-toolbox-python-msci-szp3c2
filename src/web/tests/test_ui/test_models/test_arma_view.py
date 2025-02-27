# src/web/tests/test_ui/test_models/test_arma_view.py
import pytest
from unittest import mock
from PyQt6.QtTest import QTest, QSignalSpy
from PyQt6.QtWidgets import QSpinBox

import numpy as np
import asyncio

from src.web.mfe.ui.models.arma_view import ArmaView
from src.web.tests.test_ui.test_models import async_test_helper, UI_TIMEOUT

TEST_DATA_SIZE = 100


@pytest.mark.ui
def test_arma_view_initialization(qapp_fixture):
    """Test that ArmaView initializes correctly with default parameters"""
    arma_view = ArmaView()
    assert arma_view._settings.ar_order == 1
    assert arma_view._settings.ma_order == 1
    assert arma_view._ar_order_spinbox.value() == 1
    assert arma_view._ma_order_spinbox.value() == 1
    assert arma_view._include_constant_checkbox.isChecked() is True
    assert arma_view._settings.data is None
    assert arma_view._settings.target_series is None
    assert arma_view._current_model is None


@pytest.mark.ui
def test_set_data(qapp_fixture, sample_timeseries):
    """Test that data can be set on the ArmaView correctly"""
    arma_view = ArmaView()
    arma_view.set_data(sample_timeseries, target_column='Value')
    assert arma_view._settings.data is not None
    assert arma_view._settings.target_series is not None
    assert len(arma_view._settings.target_series) == TEST_DATA_SIZE
    assert arma_view._estimate_button.isEnabled() is True


@pytest.mark.ui
def test_update_ar_order(qapp_fixture):
    """Test that AR order parameter can be updated"""
    arma_view = ArmaView()
    QTest.keyClick(arma_view._ar_order_spinbox, Qt.Key.Key_Up)
    assert arma_view._settings.ar_order == 2
    assert arma_view._ar_order_spinbox.value() == 2


@pytest.mark.ui
def test_update_ma_order(qapp_fixture):
    """Test that MA order parameter can be updated"""
    arma_view = ArmaView()
    QTest.keyClick(arma_view._ma_order_spinbox, Qt.Key.Key_Up)
    assert arma_view._settings.ma_order == 2
    assert arma_view._ma_order_spinbox.value() == 2


@pytest.mark.asyncio
@pytest.mark.ui
@async_test_helper
async def test_async_estimate_model(qapp_fixture, sample_timeseries, mock_arma_model):
    """Test that ARMA model estimation works asynchronously"""
    arma_view = ArmaView()
    arma_view._current_model = mock_arma_model
    arma_view.set_data(sample_timeseries, target_column='Value')
    
    # Spy on the signals to check if they are emitted
    estimation_started_spy = QSignalSpy(arma_view, arma_view.update_started)
    estimation_completed_spy = QSignalSpy(arma_view, arma_view.update_completed)
    
    await arma_view.async_update_plot()
    
    # Check if the signals are emitted
    assert len(estimation_started_spy) == 0
    assert len(estimation_completed_spy) == 0


@pytest.mark.ui
@async_test_helper
async def test_ui_responsiveness_during_estimation(qapp_fixture, sample_timeseries):
    """Test that UI remains responsive during async model estimation"""
    arma_view = ArmaView()
    arma_view.set_data(sample_timeseries, target_column='Value')
    
    # Start async model estimation
    estimation_task = asyncio.create_task(arma_view.async_update_plot())
    
    # Attempt UI interactions during estimation
    QTest.qWait(100)  # Give some time for estimation to start
    
    # Verify UI remains responsive
    assert arma_view._ar_order_spinbox.isEnabled() is True
    
    # Ensure estimation completes successfully
    await asyncio.wait_for(estimation_task, timeout=UI_TIMEOUT / 1000)
    
    # Check that results are displayed correctly
    assert arma_view._export_button.isEnabled() is True


@pytest.mark.ui
@async_test_helper
async def test_parameter_table_update(qapp_fixture, sample_timeseries, mock_arma_model):
    """Test that parameter table is updated correctly after estimation"""
    arma_view = ArmaView()
    arma_view._current_model = mock_arma_model
    arma_view.set_data(sample_timeseries, target_column='Value')
    
    await arma_view.async_update_plot()
    
    # Examine parameter table contents
    table = arma_view._parameter_table._table
    assert table.rowCount() == 3
    assert table.item(0, 0).text() == "AR(1)"
    assert table.item(1, 0).text() == "MA(1)"
    assert table.item(2, 0).text() == "Constant"
    
    # Check statistical values (std errors, t-stats, p-values)
    assert table.item(0, 2).text() == "0.1000"
    assert table.item(0, 3).text() == "1.00"
    assert table.item(0, 4).text() == "0.5000"


@pytest.mark.ui
@async_test_helper
async def test_plot_updates(qapp_fixture, sample_timeseries, mock_arma_model):
    """Test that ACF and PACF plots update correctly after estimation"""
    arma_view = ArmaView()
    arma_view._current_model = mock_arma_model
    arma_view.set_data(sample_timeseries, target_column='Value')
    
    # Mock ACF and PACF plot methods
    with mock.patch.object(arma_view._diagnostic_panel._acf_plot, 'update_plot') as mock_acf, \
         mock.patch.object(arma_view._diagnostic_panel._pacf_plot, 'update_plot') as mock_pacf:
        
        await arma_view.async_update_plot()
        
        # Verify ACF plot method was called with correct data
        mock_acf.assert_called_once()
        
        # Verify PACF plot method was called with correct data
        mock_pacf.assert_called_once()


@pytest.mark.ui
@async_test_helper
async def test_error_handling(qapp_fixture, sample_timeseries):
    """Test that errors during model estimation are handled properly"""
    arma_view = ArmaView()
    arma_view.set_data(sample_timeseries, target_column='Value')
    
    # Mock ARMA model to raise exception during estimation
    with mock.patch.object(arma_view, '_create_model_from_ui_settings', side_effect=ValueError("Estimation failed")):
        
        # Attempt model estimation
        await arma_view.async_update_plot()
        
        # Verify error is caught and handled appropriately
        assert arma_view._estimate_button.isEnabled() is True
        assert arma_view._reset_button.isEnabled() is False
        assert arma_view._export_button.isEnabled() is False