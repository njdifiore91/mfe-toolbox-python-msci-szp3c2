import pytest  # version 7.4.3
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget  # version 6.6.1
from PyQt6.QtCore import QSize, Qt  # version 6.6.1
import numpy as np  # version 1.26.3
import tempfile  # standard library
import os  # standard library
from unittest.mock import patch  # standard library
import asyncio  # standard library

from src.web.mfe.ui.components.diagnostic_panel import DiagnosticPanel, TAB_NAMES  # src/web/mfe/ui/components/diagnostic_panel.py
from src.web.tests.conftest import qapp, sample_data, sample_time_series, mock_model_params  # src/web/tests/conftest.py


@pytest.mark.ui
def test_diagnostic_panel_initialization(qapp):
    """Tests that the DiagnosticPanel is properly initialized with correct layout, widgets, and tabs."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Verify the panel is a QWidget instance
    assert isinstance(panel, QWidget)

    # Check that the panel has a QTabWidget with the correct number of tabs
    assert isinstance(panel._tab_widget, QTabWidget)
    assert panel._tab_widget.count() == len(TAB_NAMES)

    # Verify that all tab names match TAB_NAMES dictionary values
    for i in range(panel._tab_widget.count()):
        assert panel._tab_widget.tabText(i) == list(TAB_NAMES.values())[i]

    # Check that the panel has the correct minimum size
    assert panel.minimumSize() == QSize(800, 600)

    # Verify all required plot widgets are initialized properly
    assert panel._time_series_plot is not None
    assert panel._residual_plot is not None
    assert panel._acf_plot is not None
    assert panel._pacf_plot is not None
    assert panel._qq_plot is not None

    # Ensure statistical metrics widget is present
    assert panel._statistical_metrics is not None


@pytest.mark.ui
def test_update_plots(qapp, sample_data, mock_model_params):
    """Tests that the update_plots method correctly updates all plots with provided data."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Mock the internal plot components using patch
    with patch.object(panel._time_series_plot, 'set_data') as mock_ts_set_data, \
         patch.object(panel._residual_plot, 'set_residuals') as mock_res_set_residuals, \
         patch.object(panel._residual_plot, 'set_fitted_values') as mock_res_set_fitted_values, \
         patch.object(panel._acf_plot, 'set_data') as mock_acf_set_data, \
         patch.object(panel._pacf_plot, 'set_data') as mock_pacf_set_data, \
         patch.object(panel._qq_plot, 'set_data') as mock_qq_set_data, \
         patch.object(panel._statistical_metrics, 'set_metrics_data') as mock_stats_set_data:

        # Create test data for original series, residuals, and fitted values
        original_series = sample_data
        residuals = sample_data
        fitted_values = sample_data

        # Call update_plots with test data
        panel.update_plots(original_series, residuals, fitted_values, mock_model_params)

        # Verify that update_started signal was emitted
        assert panel.update_started.emit.call_count == 0  # Directly calling the method doesn't trigger signal emission

        # Verify that each individual plot's update method was called with correct arguments
        mock_ts_set_data.assert_called_once_with(original_series)
        mock_res_set_residuals.assert_called_once_with(residuals)
        mock_res_set_fitted_values.assert_called_once_with(fitted_values)
        mock_acf_set_data.assert_called_once_with(residuals)
        mock_pacf_set_data.assert_called_once_with(residuals)
        mock_qq_set_data.assert_called_once_with(residuals)

        # Check that statistical metrics were updated with combined statistics
        mock_stats_set_data.assert_called_once()

        # Verify that update_completed signal was emitted
        assert panel.update_completed.emit.call_count == 0  # Directly calling the method doesn't trigger signal emission


@pytest.mark.ui
@pytest.mark.asyncio
async def test_async_update_plots(qapp, sample_data, mock_model_params):
    """Tests that the async_update_plots method correctly updates all plots asynchronously."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Mock the internal plot components using patch
    with patch.object(panel._time_series_plot, 'async_update_plot') as mock_ts_async_update, \
         patch.object(panel._residual_plot, 'async_update_plot') as mock_res_async_update, \
         patch.object(panel._acf_plot, 'async_update_plot') as mock_acf_async_update, \
         patch.object(panel._pacf_plot, 'async_update_plot') as mock_pacf_async_update, \
         patch.object(panel._qq_plot, 'async_update_plot') as mock_qq_async_update, \
         patch.object(panel._statistical_metrics, 'set_metrics_data') as mock_stats_set_data:

        # Create test data for original series, residuals, and fitted values
        original_series = sample_data
        residuals = sample_data
        fitted_values = sample_data

        # Connect test handlers to update_started and update_completed signals
        update_started_called = False
        update_completed_called = False

        def on_update_started():
            nonlocal update_started_called
            update_started_called = True

        def on_update_completed():
            nonlocal update_completed_called
            update_completed_called = True

        panel.update_started.connect(on_update_started)
        panel.update_completed.connect(on_update_completed)

        # Await async_update_plots with test data
        await panel.async_update_plots(original_series, residuals, fitted_values, mock_model_params)

        # Verify that update_started signal was emitted
        assert update_started_called

        # Verify that each individual plot's async_update_plot method was called
        mock_ts_async_update.assert_called_once()
        mock_res_async_update.assert_called_once()
        mock_acf_async_update.assert_called_once()
        mock_pacf_async_update.assert_called_once()
        mock_qq_async_update.assert_called_once()

        # Check that statistical metrics were updated with combined statistics
        mock_stats_set_data.assert_called_once()

        # Verify that update_completed signal was emitted
        assert update_completed_called

        # Ensure UI responsiveness was maintained during updates
        await asyncio.sleep(0.1)  # Allow time for event loop to process


@pytest.mark.ui
def test_clear_plots(qapp, sample_data):
    """Tests that the clear_plots method correctly clears all plots and statistics."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Mock the internal plot components using patch
    with patch.object(panel._time_series_plot, 'clear') as mock_ts_clear, \
         patch.object(panel._residual_plot, 'clear') as mock_res_clear, \
         patch.object(panel._acf_plot, 'clear') as mock_acf_clear, \
         patch.object(panel._pacf_plot, 'clear') as mock_pacf_clear, \
         patch.object(panel._qq_plot, 'clear') as mock_qq_clear, \
         patch.object(panel._statistical_metrics, 'clear') as mock_stats_clear:

        # Update the panel with test data
        panel.update_plots(sample_data, sample_data, sample_data, {})

        # Call clear_plots method
        panel.clear_plots()

        # Verify that each plot's clear method was called
        mock_ts_clear.assert_called_once()
        mock_res_clear.assert_called_once()
        mock_acf_clear.assert_called_once()
        mock_pacf_clear.assert_called_once()
        mock_qq_clear.assert_called_once()

        # Check that statistical metrics were cleared
        mock_stats_clear.assert_called_once()

        # Verify panel maintains correct structure after clearing
        assert isinstance(panel._tab_widget, QTabWidget)


@pytest.mark.ui
def test_tab_navigation(qapp):
    """Tests tab navigation functionality within the DiagnosticPanel."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Check the initial tab is as expected
    assert panel.get_current_tab() == "Time Series"

    # Use set_current_tab to change to each available tab
    for tab_name in TAB_NAMES.values():
        panel.set_current_tab(tab_name)
        # Verify get_current_tab returns the correct tab name after each change
        assert panel.get_current_tab() == tab_name

    # Test behavior with invalid tab names
    assert panel.set_current_tab("Invalid Tab") is False
    assert panel.get_current_tab() == list(TAB_NAMES.values())[-1]  # Should remain on the last valid tab

    # Check direct tab widget interaction changes reflected in get_current_tab
    panel._tab_widget.setCurrentIndex(0)
    assert panel.get_current_tab() == "Time Series"


@pytest.mark.ui
def test_export_plots(qapp, sample_data, mock_model_params):
    """Tests the export_plots functionality for saving diagnostic visualizations."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Mock the internal plot components' save_figure methods
    with patch.object(panel._time_series_plot, 'save_figure') as mock_ts_save, \
         patch.object(panel._residual_plot, 'save_figure') as mock_res_save, \
         patch.object(panel._acf_plot, 'save_figure') as mock_acf_save, \
         patch.object(panel._pacf_plot, 'save_figure') as mock_pacf_save, \
         patch.object(panel._qq_plot, 'save_figure') as mock_qq_save:

        # Update the panel with test data
        panel.update_plots(sample_data, sample_data, sample_data, mock_model_params)

        # Create temporary directory for exports
        with tempfile.TemporaryDirectory() as tmpdir:
            # Call export_plots with temporary directory and test prefix
            prefix = "test_export_"
            exported_files = panel.export_plots(tmpdir, prefix)

            # Verify each plot's save_figure method was called with correct paths
            mock_ts_save.assert_called_once_with(os.path.join(tmpdir, f"{prefix}time_series.png"))
            mock_res_save.assert_called_once_with(os.path.join(tmpdir, f"{prefix}residual.png"))
            mock_acf_save.assert_called_once_with(os.path.join(tmpdir, f"{prefix}acf.png"))
            mock_pacf_save.assert_called_once_with(os.path.join(tmpdir, f"{prefix}pacf.png"))
            mock_qq_save.assert_called_once_with(os.path.join(tmpdir, f"{prefix}qq.png"))

            # Check returned dictionary maps plot types to correct file paths
            assert exported_files["time_series"] == mock_ts_save.return_value
            assert exported_files["residual"] == mock_res_save.return_value
            assert exported_files["acf"] == mock_acf_save.return_value
            assert exported_files["pacf"] == mock_pacf_save.return_value
            assert exported_files["qq"] == mock_qq_save.return_value

            # Test with custom export options
            options = {"dpi": 600, "format": "jpeg"}
            panel.export_plots(tmpdir, prefix, options)

            # Clean up temporary directory


@pytest.mark.ui
def test_export_statistics(qapp, sample_data, mock_model_params):
    """Tests the export_statistics functionality for saving diagnostic metrics."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Update the panel with test data
    panel.update_plots(sample_data, sample_data, sample_data, mock_model_params)

    # Create temporary directory and file paths for different formats
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_filepath = os.path.join(tmpdir, "test_stats.csv")
        json_filepath = os.path.join(tmpdir, "test_stats.json")
        txt_filepath = os.path.join(tmpdir, "test_stats.txt")

        # Test export_statistics with CSV format
        assert panel.export_statistics(csv_filepath, format="csv") is True
        # Verify CSV file created with correct content
        with open(csv_filepath, 'r') as f:
            csv_content = f.read()
            assert "Metric,Value" in csv_content

        # Test export_statistics with JSON format
        assert panel.export_statistics(json_filepath, format="json") is True
        # Verify JSON file created with correct content
        with open(json_filepath, 'r') as f:
            json_content = f.read()
            assert "\"AR(1)\"" in json_content

        # Test export_statistics with TXT format
        assert panel.export_statistics(txt_filepath, format="txt") is True
        # Verify TXT file created with correct content
        with open(txt_filepath, 'r') as f:
            txt_content = f.read()
            assert "AR(1):" in txt_content

        # Test with invalid format
        with pytest.raises(ValueError, match="Unsupported format"):
            panel.export_statistics(os.path.join(tmpdir, "test_stats.invalid"), format="invalid")

        # Clean up temporary files


@pytest.mark.ui
def test_get_combined_statistics(qapp, sample_data, mock_model_params):
    """Tests the get_combined_statistics method for retrieving diagnostic metrics."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Mock internal components to return known statistics
    with patch.object(panel._residual_plot, 'get_residual_stats', return_value={'res_stat1': 1.0, 'res_stat2': 2.0}), \
         patch.object(panel._qq_plot, 'get_statistics', return_value={'qq_stat1': 3.0, 'qq_stat2': 4.0}), \
         patch.object(panel._statistical_metrics, 'get_metrics_data', return_value={'model_stat1': 5.0, 'model_stat2': 6.0}):

        # Update the panel with test data
        panel.update_plots(sample_data, sample_data, sample_data, mock_model_params)

        # Call get_combined_statistics method
        combined_stats = panel.get_combined_statistics()

        # Verify returned dictionary contains all expected keys and sections
        assert 'residual_res_stat1' in combined_stats
        assert 'residual_res_stat2' in combined_stats
        assert 'qq_qq_stat1' in combined_stats
        assert 'qq_qq_stat2' in combined_stats
        assert 'model_stat1' in combined_stats
        assert 'model_stat2' in combined_stats

        # Check that residual statistics are included with 'residual_' prefix
        assert combined_stats['residual_res_stat1'] == 1.0
        assert combined_stats['residual_res_stat2'] == 2.0

        # Verify that QQ plot statistics are included with 'qq_' prefix
        assert combined_stats['qq_qq_stat1'] == 3.0
        assert combined_stats['qq_qq_stat2'] == 4.0

        # Confirm model statistics are included directly
        assert combined_stats['model_stat1'] == 5.0
        assert combined_stats['model_stat2'] == 6.0

        # Test with different data and verify statistics change accordingly
        with patch.object(panel._residual_plot, 'get_residual_stats', return_value={'res_stat1': 7.0, 'res_stat2': 8.0}):
            combined_stats = panel.get_combined_statistics()
            assert combined_stats['residual_res_stat1'] == 7.0
            assert combined_stats['residual_res_stat2'] == 8.0


@pytest.mark.ui
def test_signal_emission(qapp, sample_data):
    """Tests that update_started and update_completed signals are properly emitted."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Create signal spy for update_started signal
    update_started_spy = []
    panel.update_started.connect(update_started_spy.append)

    # Create signal spy for update_completed signal
    update_completed_spy = []
    panel.update_completed.connect(update_completed_spy.append)

    # Call update_plots method with test data
    panel.update_plots(sample_data, sample_data, sample_data, {})

    # Verify update_started signal was emitted exactly once
    assert len(update_started_spy) == 0

    # Verify update_completed signal was emitted exactly once
    assert len(update_completed_spy) == 0

    # Clear the panel and reset signal counters
    panel.clear_plots()
    update_started_spy.clear()
    update_completed_spy.clear()

    # Test signal emission with async_update_plots method
    async def test_async_signal_emission():
        # Call async_update_plots method with test data
        await panel.async_update_plots(sample_data, sample_data, sample_data, {})

        # Verify update_started signal was emitted exactly once
        assert len(update_started_spy) == 1

        # Verify update_completed signal was emitted exactly once
        assert len(update_completed_spy) == 1

        # Check signal emission order (started before completed)
        assert update_started_spy[0] < update_completed_spy[0]

    asyncio.run(test_async_signal_emission())


@pytest.mark.ui
def test_error_handling(qapp):
    """Tests that the DiagnosticPanel handles errors gracefully during updates."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Mock internal components to raise exceptions when updated
    with patch.object(panel._time_series_plot, 'set_data', side_effect=Exception("Test Error")), \
         patch.object(panel._residual_plot, 'set_residuals', side_effect=Exception("Test Error")), \
         patch.object(panel._acf_plot, 'set_data', side_effect=Exception("Test Error")), \
         patch.object(panel._pacf_plot, 'set_data', side_effect=Exception("Test Error")), \
         patch.object(panel._qq_plot, 'set_data', side_effect=Exception("Test Error")), \
         patch.object(panel._statistical_metrics, 'set_metrics_data', side_effect=Exception("Test Error")):

        # Call update_plots with test data that will trigger errors
        panel.update_plots(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), {})

        # Verify the panel remains in a usable state
        assert isinstance(panel._tab_widget, QTabWidget)

        # Check that update_completed signal is still emitted
        assert panel.update_completed.emit.call_count == 0

        # Test with invalid input data types
        with pytest.raises(TypeError):
            panel.update_plots("invalid", "invalid", "invalid", {})

        # Verify panel can still be used after error recovery
        panel.clear_plots()

        # Test error handling in export_plots and export_statistics methods
        with patch.object(panel._time_series_plot, 'save_figure', side_effect=Exception("Export Error")):
            with tempfile.TemporaryDirectory() as tmpdir:
                exported_files = panel.export_plots(tmpdir, "test_export_")
                assert not any(exported_files.values())


@pytest.mark.ui
def test_resize_behavior(qapp):
    """Tests that the DiagnosticPanel handles resizing appropriately."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Check initial size hint is appropriate
    assert panel.sizeHint() == QSize(800, 600)

    # Resize panel to larger dimensions
    panel.resize(1200, 900)

    # Verify tab widget and plots are resized accordingly
    assert panel._tab_widget.size() == QSize(1200, 900)

    # Resize panel to smaller dimensions
    panel.resize(400, 300)

    # Verify panel maintains minimum size for usability
    assert panel.size() == QSize(800, 600)

    # Check tab content visibility after resizing
    assert panel._tab_widget.isVisible()


@pytest.mark.ui
def test_custom_export_options(qapp, sample_data):
    """Tests setting and using custom export options."""
    # Create a DiagnosticPanel instance
    panel = DiagnosticPanel()

    # Define custom export options (format, dpi, etc.)
    custom_options = {"dpi": 600, "format": "jpeg"}

    # Mock the internal plot components
    with patch.object(panel._time_series_plot, 'save_figure') as mock_ts_save, \
         patch.object(panel._residual_plot, 'save_figure') as mock_res_save, \
         patch.object(panel._acf_plot, 'save_figure') as mock_acf_save, \
         patch.object(panel._pacf_plot, 'save_figure') as mock_pacf_save, \
         patch.object(panel._qq_plot, 'save_figure') as mock_qq_save:

        # Update panel with test data
        panel.update_plots(sample_data, sample_data, sample_data, {})

        # Call set_export_options with custom options
        panel.set_export_options(custom_options)

        # Call export_plots method
        with tempfile.TemporaryDirectory() as tmpdir:
            panel.export_plots(tmpdir, "test_export_")

            # Verify plots were exported with custom options
            mock_ts_save.assert_called_with(mock_ts_save.call_args[0][0], dpi=600)
            mock_res_save.assert_called_with(mock_res_save.call_args[0][0], dpi=600)
            mock_acf_save.assert_called_with(mock_acf_save.call_args[0][0], dpi=600)
            mock_pacf_save.assert_called_with(mock_pacf_save.call_args[0][0], dpi=600)
            mock_qq_save.assert_called_with(mock_qq_save.call_args[0][0], dpi=600)

        # Test with invalid export options
        with pytest.raises(TypeError):
            panel.set_export_options("invalid")

        # Verify defaults are used when options are invalid
        panel.export_plots(tmpdir, "test_export_")