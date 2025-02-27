#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for the ARMAXViewer UI component.

This module implements comprehensive test cases for the ARMAX model results viewer
dialog, including initialization, display updating, navigation functionality, and
asynchronous operations.
"""

import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import MagicMock, patch
import asyncio

from PyQt6.QtTest import QTest, QSignalSpy
from PyQt6.QtCore import Qt

from mfe.ui.armax_viewer import ARMAXViewer
from tests.conftest import qapp, mock_model_params


def setup_function():
    """Setup function that runs before each test to prepare the testing environment."""
    # Reset any global state that might affect tests
    pass


def teardown_function():
    """Teardown function that runs after each test to clean up the testing environment."""
    # Clean up any resources created during tests
    pass


def create_mock_results(num_results=1):
    """
    Creates mock ARMAX model results for testing.
    
    Args:
        num_results: Number of mock results to create
        
    Returns:
        List of mock model result objects
    """
    results = []
    
    for i in range(num_results):
        # Create a mock result object
        result = MagicMock()
        
        # Set attributes
        result.ar_order = 1
        result.ma_order = 1
        result.include_constant = True
        
        # Set parameters
        result.param_names = [f'phi_{j}' for j in range(1, result.ar_order + 1)] + \
                             [f'theta_{j}' for j in range(1, result.ma_order + 1)] + \
                             (['constant'] if result.include_constant else [])
        
        result.params = np.array([0.7 + i*0.05, -0.2 - i*0.05, 0.001 + i*0.001])
        result.bse = np.array([0.05, 0.06, 0.001])
        result.tvalues = result.params / result.bse
        result.pvalues = np.array([0.001, 0.002, 0.05])
        
        # Set likelihood statistics
        result.loglike = -240.0 - i*5
        result.aic = -2.3 - i*0.1
        result.bic = -2.2 - i*0.1
        result.hqic = -2.25 - i*0.1
        result.rsquared = 0.85 - i*0.05
        
        # Set residuals and fitted values
        result.resid = np.random.randn(200)
        result.fittedvalues = np.random.randn(200)
        
        # Add diagnostic tests
        result.diagnostic_tests = {
            'ljung_box': (15.5, 0.2),
            'jarque_bera': (2.3, 0.3),
            'het_white': (1.2, 0.8),
            'serial_corr': (0.5, 0.95)
        }
        
        results.append(result)
    
    return results


@pytest.mark.ui
def test_armax_viewer_init(qapp):
    """Tests the initialization of the ARMAXViewer dialog."""
    # Create mock results
    mock_results = create_mock_results(2)
    
    # Initialize the viewer
    viewer = ARMAXViewer(mock_results)
    
    # Verify attributes are set correctly
    assert viewer._results == mock_results
    assert viewer._current_page == 0
    assert viewer._total_pages == 2
    
    # Check that UI components are initialized
    assert hasattr(viewer, '_equation_widget')
    assert hasattr(viewer, '_parameter_table')
    assert hasattr(viewer, '_statistics_widget')
    assert hasattr(viewer, '_residual_plot')
    assert hasattr(viewer, '_navigation_bar')
    
    # Clean up
    viewer.close()


@pytest.mark.ui
def test_update_display(qapp, mock_model_params):
    """Tests the update_display method of ARMAXViewer."""
    # Create mock results
    mock_results = create_mock_results(1)
    
    # Initialize the viewer
    viewer = ARMAXViewer(mock_results)
    
    # Mock the component methods to verify they're called
    viewer._equation_widget.set_arma_equation = MagicMock()
    viewer._parameter_table.set_parameter_data = MagicMock()
    viewer._statistics_widget.set_metrics_data = MagicMock()
    viewer._residual_plot.set_residuals = MagicMock()
    viewer._residual_plot.set_fitted_values = MagicMock()
    viewer._residual_plot.update_plot = MagicMock()
    viewer._navigation_bar.update_page_indicator = MagicMock()
    
    # Call update_display
    viewer.update_display()
    
    # Verify that component methods were called with appropriate arguments
    viewer._equation_widget.set_arma_equation.assert_called_once()
    viewer._parameter_table.set_parameter_data.assert_called_once()
    viewer._statistics_widget.set_metrics_data.assert_called_once()
    viewer._residual_plot.set_residuals.assert_called_once()
    viewer._residual_plot.set_fitted_values.assert_called_once()
    viewer._residual_plot.update_plot.assert_called_once()
    viewer._navigation_bar.update_page_indicator.assert_called_once_with(1, 1)
    
    # Clean up
    viewer.close()


@pytest.mark.asyncio
@pytest.mark.ui
async def test_async_update_display(qapp):
    """Tests the asynchronous display updating of ARMAXViewer."""
    # Create mock results
    mock_results = create_mock_results(1)
    
    # Initialize the viewer
    viewer = ARMAXViewer(mock_results)
    
    # Mock the component methods to verify they're called
    viewer._equation_widget.set_arma_equation = MagicMock()
    viewer._parameter_table.set_parameter_data = MagicMock()
    viewer._statistics_widget.set_metrics_data = MagicMock()
    viewer._residual_plot.set_residuals = MagicMock()
    viewer._residual_plot.set_fitted_values = MagicMock()
    viewer._residual_plot.async_update_plot = MagicMock(
        return_value=asyncio.Future()
    )
    viewer._residual_plot.async_update_plot.return_value.set_result(None)
    viewer._navigation_bar.update_page_indicator = MagicMock()
    
    # Call async_update_display
    await viewer.update_display_async()
    
    # Verify that component methods were called with appropriate arguments
    viewer._equation_widget.set_arma_equation.assert_called_once()
    viewer._parameter_table.set_parameter_data.assert_called_once()
    viewer._statistics_widget.set_metrics_data.assert_called_once()
    viewer._residual_plot.set_residuals.assert_called_once()
    viewer._residual_plot.set_fitted_values.assert_called_once()
    viewer._residual_plot.async_update_plot.assert_called_once()
    viewer._navigation_bar.update_page_indicator.assert_called_once_with(1, 1)
    
    # Clean up
    viewer.close()


@pytest.mark.ui
def test_navigation(qapp):
    """Tests the navigation between results pages in ARMAXViewer."""
    # Create mock results with multiple pages
    mock_results = create_mock_results(3)
    
    # Initialize the viewer
    viewer = ARMAXViewer(mock_results)
    
    # Mock the update_display method
    viewer.update_display = MagicMock()
    
    # Check initial page
    assert viewer._current_page == 0
    
    # Test next navigation
    viewer.on_next()
    assert viewer._current_page == 1
    viewer.update_display.assert_called_once()
    viewer.update_display.reset_mock()
    
    # Test next navigation again
    viewer.on_next()
    assert viewer._current_page == 2
    viewer.update_display.assert_called_once()
    viewer.update_display.reset_mock()
    
    # Test next navigation at last page (should not change)
    viewer.on_next()
    assert viewer._current_page == 2  # Should not change
    viewer.update_display.assert_not_called()
    
    # Test previous navigation
    viewer.on_previous()
    assert viewer._current_page == 1
    viewer.update_display.assert_called_once()
    viewer.update_display.reset_mock()
    
    # Test previous navigation again
    viewer.on_previous()
    assert viewer._current_page == 0
    viewer.update_display.assert_called_once()
    viewer.update_display.reset_mock()
    
    # Test previous navigation at first page (should not change)
    viewer.on_previous()
    assert viewer._current_page == 0  # Should not change
    viewer.update_display.assert_not_called()
    
    # Clean up
    viewer.close()


@pytest.mark.ui
def test_navigation_buttons(qapp):
    """Tests the navigation button connections in ARMAXViewer."""
    # Create mock results with multiple pages
    mock_results = create_mock_results(3)
    
    # Initialize the viewer
    viewer = ARMAXViewer(mock_results)
    
    # Mock the navigation methods
    viewer.on_next = MagicMock()
    viewer.on_previous = MagicMock()
    
    # Create signal spies
    next_spy = QSignalSpy(viewer._navigation_bar.navigated_next)
    prev_spy = QSignalSpy(viewer._navigation_bar.navigated_previous)
    
    # Simulate clicking the next button
    QTest.mouseClick(viewer._navigation_bar._next_button, Qt.MouseButton.LeftButton)
    assert next_spy.count() == 1
    viewer.on_next.assert_called_once()
    viewer.on_next.reset_mock()
    
    # Simulate clicking the previous button
    QTest.mouseClick(viewer._navigation_bar._previous_button, Qt.MouseButton.LeftButton)
    assert prev_spy.count() == 1
    viewer.on_previous.assert_called_once()
    
    # Clean up
    viewer.close()


@pytest.mark.ui
def test_close_button(qapp):
    """Tests the close button functionality of ARMAXViewer."""
    # Create mock results
    mock_results = create_mock_results(1)
    
    # Initialize the viewer
    viewer = ARMAXViewer(mock_results)
    
    # Mock the closeEvent method
    viewer.closeEvent = MagicMock()
    
    # Simulate clicking the close button
    QTest.mouseClick(viewer._close_button, Qt.MouseButton.LeftButton)
    
    # Verify that the close button is connected to the close slot
    assert viewer._close_button.receivers(viewer._close_button.clicked) > 0
    
    # Clean up
    viewer.close()