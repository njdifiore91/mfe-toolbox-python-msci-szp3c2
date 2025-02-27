"""
PyQt6-based widget component that displays model estimation results in a formatted table,
showing parameter estimates, standard errors, t-statistics, and p-values with configurable
significance highlighting.
"""

# Internal imports
from ..async.signals import ModelSignals

# External imports
from PyQt6.QtWidgets import (
    QWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QFont, QColor, QBrush
from PyQt6.QtCore import Qt, pyqtSlot
import typing
import logging
import numpy as np
import pandas as pd
import csv

# Module logger
logger = logging.getLogger(__name__)

class ParameterTable(QWidget):
    """
    A PyQt6 table widget that displays model parameter estimates with formatting and significance highlighting.
    """
    
    def __init__(self, parent: typing.Optional[QWidget] = None):
        """
        Initializes the parameter table widget with headers and default settings.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Default settings
        self.significance_level = 0.05
        self.highlight_significant = True
        
        # Initialize table headers and data
        self._headers = ["Parameter", "Estimate", "Std.Error", "t-stat", "p-value"]
        self._parameter_data = {}
        
        # Create fonts and colors for styling
        self.bold_font = QFont()
        self.bold_font.setBold(True)
        
        self.highlight_color = QColor(220, 240, 255)  # Light blue
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """
        Sets up the UI components for the parameter table.
        """
        # Create main layout
        layout = QVBoxLayout()
        
        # Create table widget
        self._table = QTableWidget()
        self._table.setColumnCount(len(self._headers))
        self._table.setHorizontalHeaderLabels(self._headers)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, len(self._headers)):
            self._table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        
        layout.addWidget(self._table)
        
        # Create export button
        export_layout = QHBoxLayout()
        export_button = QPushButton("Export to CSV")
        export_button.clicked.connect(self.export_table)
        export_layout.addStretch()
        export_layout.addWidget(export_button)
        
        layout.addLayout(export_layout)
        
        self.setLayout(layout)
    
    def connect_signals(self, model_signals: ModelSignals):
        """
        Connects this widget to a ModelSignals instance to receive parameter updates.
        
        Args:
            model_signals: ModelSignals instance to connect to
        """
        model_signals.parameters_updated.connect(self._handle_parameters_updated)
        logger.debug("Connected to model signals")
    
    def set_parameter_data(self, parameter_data: dict):
        """
        Populates the table with parameter estimation results.
        
        Args:
            parameter_data: Dictionary containing parameter estimation results
        """
        self._parameter_data = parameter_data
        
        # Clear existing table data
        self._table.clearContents()
        
        # Set row count
        param_count = len(parameter_data)
        self._table.setRowCount(param_count)
        
        # Populate table with data
        for row, (param_name, param_values) in enumerate(parameter_data.items()):
            # Parameter name
            self._table.setItem(row, 0, QTableWidgetItem(param_name))
            
            # Parameter values (estimate, std err, t-stat, p-value)
            for col, value_type in enumerate(['estimate', 'std_error', 't_stat', 'p_value']):
                if value_type in param_values:
                    value = param_values[value_type]
                    formatted_value = self._format_value(value, col + 1)
                    self._table.setItem(row, col + 1, QTableWidgetItem(formatted_value))
        
        # Apply styling
        self._apply_styling()
        
        # Resize columns to content
        self._table.resizeColumnsToContents()
        
        logger.debug(f"Parameter table updated with {param_count} parameters")
    
    def _format_value(self, value: typing.Any, column_index: int) -> str:
        """
        Formats different types of values appropriately for display.
        
        Args:
            value: The value to format
            column_index: The column index (determines formatting style)
            
        Returns:
            Formatted string representation
        """
        if value is None:
            return "-"
        
        # Parameter name
        if column_index == 0:
            return str(value)
        
        # Estimate and Std.Error
        if column_index in [1, 2]:
            return f"{value:.4f}"
        
        # t-stat
        if column_index == 3:
            return f"{value:.2f}"
        
        # p-value
        if column_index == 4:
            if value < 0.001:
                return "< 0.001"
            else:
                return f"{value:.4f}"
        
        # Default formatting
        return str(value)
    
    def _apply_styling(self):
        """
        Applies visual styling to table cells including significance highlighting.
        """
        # Set text alignment (left for parameter names, right for numeric values)
        for row in range(self._table.rowCount()):
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                if item is not None:
                    # Left-align parameter names, right-align numeric columns
                    alignment = Qt.AlignmentFlag.AlignLeft if col == 0 else Qt.AlignmentFlag.AlignRight
                    item.setTextAlignment(alignment | Qt.AlignmentFlag.AlignVCenter)
        
        # Apply significance highlighting if enabled
        if self.highlight_significant:
            for row in range(self._table.rowCount()):
                p_value_item = self._table.item(row, 4)  # p-value column
                if p_value_item is not None:
                    p_value_text = p_value_item.text()
                    
                    try:
                        # Handle "< 0.001" case
                        if p_value_text.startswith("<"):
                            is_significant = True
                        else:
                            p_value = float(p_value_text)
                            is_significant = p_value < self.significance_level
                            
                        if is_significant:
                            # Highlight significant rows
                            for col in range(self._table.columnCount()):
                                item = self._table.item(row, col)
                                if item is not None:
                                    item.setFont(self.bold_font)
                                    item.setBackground(QBrush(self.highlight_color))
                    except ValueError:
                        # Skip if p-value is not a valid number
                        pass
    
    def toggle_highlight(self):
        """
        Toggles the highlighting of statistically significant parameters.
        """
        self.highlight_significant = not self.highlight_significant
        
        # Reapply styling if we have data
        if self._parameter_data:
            self._apply_styling()
            
        logger.debug(f"Parameter highlighting toggled to {self.highlight_significant}")
    
    def set_significance_level(self, alpha: float):
        """
        Sets the significance level threshold for highlighting.
        
        Args:
            alpha: Significance level (between 0 and 1)
        """
        if not 0 < alpha < 1:
            raise ValueError("Significance level must be between 0 and 1")
        
        self.significance_level = alpha
        
        # Reapply styling if we have data
        if self._parameter_data:
            self._apply_styling()
            
        logger.debug(f"Significance level set to {alpha}")
    
    def export_table(self) -> bool:
        """
        Exports the parameter table data to a CSV file.
        
        Returns:
            Success status of the export operation
        """
        if not self._parameter_data:
            QMessageBox.warning(self, "Export Error", "No data to export")
            return False
        
        # Get file save location
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Parameter Table", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            # User cancelled
            return False
            
        try:
            # Convert parameter data to DataFrame for easy export
            data = []
            for param_name, param_values in self._parameter_data.items():
                row = {
                    'Parameter': param_name,
                    'Estimate': param_values.get('estimate', None),
                    'Std.Error': param_values.get('std_error', None),
                    't-stat': param_values.get('t_stat', None),
                    'p-value': param_values.get('p_value', None)
                }
                data.append(row)
                
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            logger.info(f"Parameter table exported to {filename}")
            QMessageBox.information(self, "Export Successful", 
                                   f"Parameter table exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting parameter table: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")
            return False
    
    def clear(self):
        """
        Clears all data from the parameter table.
        """
        self._parameter_data = {}
        self._table.setRowCount(0)
        logger.debug("Parameter table cleared")
    
    @pyqtSlot(object)
    def _handle_parameters_updated(self, parameters):
        """
        Slot method that handles the parameters_updated signal.
        
        Args:
            parameters: Parameter data received from signal
        """
        # Convert parameter data if needed
        if not isinstance(parameters, dict):
            # Try to convert to appropriate format
            try:
                param_dict = {}
                # Assume parameters is a model result object with a params attribute
                if hasattr(parameters, 'params'):
                    for i, param_name in enumerate(parameters.params.index):
                        param_dict[param_name] = {
                            'estimate': parameters.params[i],
                            'std_error': parameters.bse[i] if hasattr(parameters, 'bse') else None,
                            't_stat': parameters.tvalues[i] if hasattr(parameters, 'tvalues') else None,
                            'p_value': parameters.pvalues[i] if hasattr(parameters, 'pvalues') else None
                        }
                parameters = param_dict
            except Exception as e:
                logger.error(f"Failed to process parameters: {str(e)}")
                return
        
        self.set_parameter_data(parameters)
        logger.debug("Parameters updated from signal")
    
    def get_parameter_data(self) -> dict:
        """
        Returns the current parameter data dictionary.
        
        Returns:
            Current parameter data or empty dict if none
        """
        return self._parameter_data.copy()