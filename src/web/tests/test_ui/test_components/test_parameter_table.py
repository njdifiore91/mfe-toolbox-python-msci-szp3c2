"""
Unit tests for the ParameterTable component which displays statistical model
parameter estimates and related metrics in the PyQt6-based interface.
"""

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidget
import numpy as np

# Import the component to be tested
from mfe.ui.components.parameter_table import ParameterTable


@pytest.mark.ui
def test_parameter_table_initialization(qtbot):
    """Tests that the parameter table initializes correctly with the proper columns and settings."""
    # Create the parameter table widget
    param_table = ParameterTable()
    qtbot.addWidget(param_table)
    
    # Get the QTableWidget from our component
    table_widget = param_table._table
    
    # Test column count and headers
    assert table_widget.columnCount() == 5
    expected_headers = ["Parameter", "Estimate", "Std.Error", "t-stat", "p-value"]
    for col in range(table_widget.columnCount()):
        assert table_widget.horizontalHeaderItem(col).text() == expected_headers[col]
    
    # Test that the table is initially empty
    assert table_widget.rowCount() == 0
    
    # Test table properties
    assert table_widget.alternatingRowColors() is True
    assert table_widget.selectionBehavior() == QTableWidget.SelectionBehavior.SelectRows
    assert table_widget.horizontalHeader().isVisible() is True


@pytest.mark.ui
def test_parameter_table_data_loading(qtbot):
    """Tests that the parameter table correctly loads and displays parameter data."""
    # Create the parameter table widget
    param_table = ParameterTable()
    qtbot.addWidget(param_table)
    
    # Create mock parameter data
    mock_data = {
        "AR(1)": {
            "estimate": 0.7562,
            "std_error": 0.0452,
            "t_stat": 16.7345,
            "p_value": 0.0000
        },
        "MA(1)": {
            "estimate": -0.2431,
            "std_error": 0.0674,
            "t_stat": -3.6067,
            "p_value": 0.0003
        },
        "Constant": {
            "estimate": 0.0021,
            "std_error": 0.0010,
            "t_stat": 2.0452,
            "p_value": 0.0410
        }
    }
    
    # Update the table with mock data
    param_table.set_parameter_data(mock_data)
    
    # Test that rows were created correctly
    table_widget = param_table._table
    assert table_widget.rowCount() == 3
    
    # Test cell contents
    assert table_widget.item(0, 0).text() == "AR(1)"
    assert table_widget.item(0, 1).text() == "0.7562"  # Estimate
    assert table_widget.item(0, 2).text() == "0.0452"  # Std.Error
    assert table_widget.item(0, 3).text() == "16.73"   # t-stat (formatted to 2 decimals)
    assert table_widget.item(0, 4).text() == "< 0.001" # p-value (formatted as < 0.001)
    
    assert table_widget.item(1, 0).text() == "MA(1)"
    assert table_widget.item(1, 1).text() == "-0.2431" # Estimate
    
    assert table_widget.item(2, 0).text() == "Constant"
    assert table_widget.item(2, 4).text() == "0.0410"  # p-value (formatted to 4 decimals)


@pytest.mark.ui
def test_parameter_table_formatting(qtbot):
    """Tests that the parameter table correctly formats numerical values according to statistical conventions."""
    # Create the parameter table widget
    param_table = ParameterTable()
    qtbot.addWidget(param_table)
    
    # Create mock data with specific values requiring different formatting
    mock_data = {
        "Parameter1": {
            "estimate": 0.00001234,  # Small value, should show 4 decimals
            "std_error": 0.00000567,
            "t_stat": 2.17637,
            "p_value": 0.00001       # Very small p-value, should use scientific notation
        },
        "Parameter2": {
            "estimate": 1000.5678,   # Large value
            "std_error": 100.1234,
            "t_stat": 10.001,
            "p_value": 0.051         # Just above significance threshold
        },
        "Parameter3": {
            "estimate": -0.5678,     # Negative value
            "std_error": 0.1234,
            "t_stat": -4.6012,
            "p_value": 0.031         # Significant p-value
        }
    }
    
    # Update the table with the test data
    param_table.set_parameter_data(mock_data)
    
    # Test formatting of different values
    table_widget = param_table._table
    
    # Test small value formatting
    assert table_widget.item(0, 1).text() == "0.0000"  # Rounds to 4 decimals
    
    # Test small p-value formatting
    assert table_widget.item(0, 4).text() == "< 0.001"
    
    # Test large value formatting
    assert table_widget.item(1, 1).text() == "1000.5678"
    
    # Test negative value formatting
    assert table_widget.item(2, 1).text() == "-0.5678"
    assert table_widget.item(2, 3).text() == "-4.60"


@pytest.mark.ui
def test_parameter_table_clear(qtbot):
    """Tests that the parameter table clear method removes data while preserving the structure."""
    # Create the parameter table widget
    param_table = ParameterTable()
    qtbot.addWidget(param_table)
    
    # Add mock data
    mock_data = {
        "Parameter1": {
            "estimate": 0.123,
            "std_error": 0.045,
            "t_stat": 2.73,
            "p_value": 0.006
        }
    }
    
    # Update the table with data
    param_table.set_parameter_data(mock_data)
    
    # Verify data was added
    table_widget = param_table._table
    assert table_widget.rowCount() == 1
    
    # Clear the table
    param_table.clear()
    
    # Test that data was removed
    assert table_widget.rowCount() == 0
    
    # Test that column headers are preserved
    assert table_widget.columnCount() == 5
    assert table_widget.horizontalHeaderItem(0).text() == "Parameter"
    
    # Test that we can add data again after clearing
    param_table.set_parameter_data(mock_data)
    assert table_widget.rowCount() == 1


@pytest.mark.ui
def test_parameter_table_resize(qtbot):
    """Tests that the parameter table handles resizing appropriately."""
    # Create the parameter table widget with initial size
    param_table = ParameterTable()
    param_table.resize(400, 300)
    qtbot.addWidget(param_table)
    
    # Add mock data
    mock_data = {
        "Parameter1": {"estimate": 0.123, "std_error": 0.045, "t_stat": 2.73, "p_value": 0.006},
        "Parameter2": {"estimate": 0.456, "std_error": 0.078, "t_stat": 5.85, "p_value": 0.000},
        "Parameter3": {"estimate": -0.789, "std_error": 0.101, "t_stat": -7.81, "p_value": 0.000}
    }
    param_table.set_parameter_data(mock_data)
    
    # Get initial column widths
    table_widget = param_table._table
    initial_widths = [table_widget.columnWidth(i) for i in range(table_widget.columnCount())]
    
    # Resize to larger size
    param_table.resize(600, 400)
    
    # Test that column widths adjust (first column should stretch)
    assert table_widget.columnWidth(0) > initial_widths[0]
    
    # Resize to smaller size
    param_table.resize(300, 200)
    
    # Check that content remains visible (headers should still be accessible)
    assert table_widget.horizontalHeader().isVisible()


@pytest.mark.ui
def test_parameter_table_significant_values(qtbot):
    """Tests that the parameter table properly highlights statistically significant values."""
    # Create the parameter table widget
    param_table = ParameterTable()
    qtbot.addWidget(param_table)
    
    # Set significance level
    param_table.set_significance_level(0.05)
    param_table.highlight_significant = True
    
    # Create mock data with mix of significant and non-significant parameters
    mock_data = {
        "Significant1": {"estimate": 0.789, "std_error": 0.123, "t_stat": 6.415, "p_value": 0.001},
        "Significant2": {"estimate": -0.456, "std_error": 0.098, "t_stat": -4.653, "p_value": 0.049},
        "NotSignificant1": {"estimate": 0.234, "std_error": 0.167, "t_stat": 1.401, "p_value": 0.161},
        "BorderCase": {"estimate": 0.345, "std_error": 0.176, "t_stat": 1.960, "p_value": 0.050}
    }
    
    # Update the table with the test data
    param_table.set_parameter_data(mock_data)
    
    # Get the table widget
    table_widget = param_table._table
    
    # Test that significant rows are highlighted (by checking font weight)
    assert table_widget.item(0, 0).font().bold() is True  # p=0.001 < 0.05
    assert table_widget.item(1, 0).font().bold() is True  # p=0.049 < 0.05
    
    # Test that non-significant rows are not highlighted
    assert table_widget.item(2, 0).font().bold() is False  # p=0.161 > 0.05
    
    # Test border case (p = 0.05 should not be highlighted as it's not < 0.05)
    assert table_widget.item(3, 0).font().bold() is False  # p=0.05 == 0.05
    
    # Test that toggling highlighting works
    param_table.toggle_highlight()
    assert param_table.highlight_significant is False