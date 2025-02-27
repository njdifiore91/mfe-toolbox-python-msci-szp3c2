"""
PyQt6-based UI component for displaying statistical metrics from econometric models.

This module provides a widget for displaying organized and formatted statistical metrics,
including information criteria, likelihood values, and diagnostic test results.
"""

import logging
import math
from typing import Any, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFrame, QSizePolicy
)

from ..styles import style_group_box, get_title_font, get_normal_font, get_small_font
from ..async.signals import ModelSignals

# Configure module logger
logger = logging.getLogger(__name__)


def create_metric_label(metric_name: str, metric_value: Any, is_primary: bool) -> Tuple[QLabel, QLabel]:
    """
    Creates a pair of labels for a metric name and its value with proper styling.
    
    Args:
        metric_name: The name of the metric to display
        metric_value: The value of the metric to display
        is_primary: Whether this is a primary/important metric that should be highlighted
        
    Returns:
        A tuple containing (name_label, value_label)
    """
    # Create name label
    name_label = QLabel(f"{metric_name}:")
    name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    
    # Create value label
    value_label = QLabel(format_metric_value(metric_value))
    value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    
    # Apply styling based on importance
    if is_primary:
        font = get_normal_font()
        font.setBold(True)
        name_label.setFont(font)
        value_label.setFont(font)
    else:
        name_label.setFont(get_small_font())
        value_label.setFont(get_small_font())
    
    return name_label, value_label


def format_metric_value(value: Any) -> str:
    """
    Formats a metric value for display with appropriate precision and handling of special cases.
    
    Args:
        value: The value to format
        
    Returns:
        Formatted value as string
    """
    if value is None:
        return "-"
    
    if isinstance(value, float):
        # Handle different magnitudes of floating point values
        abs_value = abs(value)
        if abs_value == 0:
            return "0.000"
        elif abs_value < 0.001:
            return f"{value:.2e}"  # Scientific notation for very small values
        elif abs_value < 1:
            return f"{value:.4f}"  # 4 decimal places for values between 0.001 and 1
        elif abs_value < 100:
            return f"{value:.3f}"  # 3 decimal places for moderate values
        else:
            return f"{value:.2f}"  # 2 decimal places for large values
    
    if isinstance(value, bool):
        return "Yes" if value else "No"
    
    # Default case: convert to string
    return str(value)


class StatisticalMetrics(QWidget):
    """
    A widget that displays statistical metrics for econometric models in an organized,
    visually appealing format.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the statistical metrics widget with grouped display sections.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Initialize data storage
        self._metrics_data = {}
        self._label_cache = {}  # Cache for value labels to easily update them
        
        # Set up UI
        self.setup_ui()
        
        # Initialize signal connector
        self.model_signals = None
        
    def setup_ui(self) -> None:
        """
        Sets up the UI components for the statistical metrics display.
        """
        # Create main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(10, 10, 10, 10)
        self._main_layout.setSpacing(10)
        
        # Create group boxes for different metric categories
        self._criteria_group = QGroupBox("Information Criteria")
        self._likelihood_group = QGroupBox("Likelihood Statistics")
        self._diagnostic_group = QGroupBox("Diagnostic Tests")
        
        # Set up layouts for each group
        criteria_layout = QHBoxLayout(self._criteria_group)
        likelihood_layout = QHBoxLayout(self._likelihood_group)
        diagnostic_layout = QVBoxLayout(self._diagnostic_group)
        
        # Information Criteria group - create dual-column layout
        criteria_left_layout = QVBoxLayout()
        criteria_right_layout = QVBoxLayout()
        criteria_layout.addLayout(criteria_left_layout)
        criteria_layout.addLayout(criteria_right_layout)
        
        # Add common information criteria
        aic_name, aic_value = create_metric_label("AIC", None, True)
        bic_name, bic_value = create_metric_label("BIC", None, True)
        hqic_name, hqic_value = create_metric_label("HQIC", None, False)
        
        criteria_left_layout.addWidget(aic_name)
        criteria_left_layout.addWidget(bic_name)
        criteria_left_layout.addWidget(hqic_name)
        
        criteria_right_layout.addWidget(aic_value)
        criteria_right_layout.addWidget(bic_value)
        criteria_right_layout.addWidget(hqic_value)
        
        # Cache value labels for easy updates
        self._label_cache['aic_value'] = aic_value
        self._label_cache['bic_value'] = bic_value
        self._label_cache['hqic_value'] = hqic_value
        
        # Likelihood Statistics group - create dual-column layout
        likelihood_left_layout = QVBoxLayout()
        likelihood_right_layout = QVBoxLayout()
        likelihood_layout.addLayout(likelihood_left_layout)
        likelihood_layout.addLayout(likelihood_right_layout)
        
        # Add common likelihood statistics
        loglike_name, loglike_value = create_metric_label("Log-Likelihood", None, True)
        ssr_name, ssr_value = create_metric_label("Sum of Squared Residuals", None, False)
        rsquared_name, rsquared_value = create_metric_label("R-squared", None, True)
        
        likelihood_left_layout.addWidget(loglike_name)
        likelihood_left_layout.addWidget(ssr_name)
        likelihood_left_layout.addWidget(rsquared_name)
        
        likelihood_right_layout.addWidget(loglike_value)
        likelihood_right_layout.addWidget(ssr_value)
        likelihood_right_layout.addWidget(rsquared_value)
        
        # Cache value labels
        self._label_cache['loglike_value'] = loglike_value
        self._label_cache['ssr_value'] = ssr_value
        self._label_cache['rsquared_value'] = rsquared_value
        
        # Diagnostic Tests group - create table-like layout
        diagnostic_grid = QVBoxLayout()
        diagnostic_layout.addLayout(diagnostic_grid)
        
        # Add test headers
        header_layout = QHBoxLayout()
        test_header = QLabel("Test")
        stat_header = QLabel("Statistic")
        pval_header = QLabel("p-value")
        
        # Set font for headers
        header_font = get_normal_font()
        header_font.setBold(True)
        test_header.setFont(header_font)
        stat_header.setFont(header_font)
        pval_header.setFont(header_font)
        
        # Configure header layout
        header_layout.addWidget(test_header, 4)
        header_layout.addWidget(stat_header, 3)
        header_layout.addWidget(pval_header, 3)
        diagnostic_grid.addLayout(header_layout)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        diagnostic_grid.addWidget(separator)
        
        # Add common diagnostic tests
        tests = ["Ljung-Box", "Jarque-Bera", "Heteroskedasticity", "Serial Correlation"]
        
        for test_name in tests:
            test_layout = QHBoxLayout()
            test_label = QLabel(test_name)
            test_label.setFont(get_small_font())
            
            stat_value = QLabel("-")
            stat_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            stat_value.setFont(get_small_font())
            
            pval_value = QLabel("-")
            pval_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pval_value.setFont(get_small_font())
            
            # Add to layout
            test_layout.addWidget(test_label, 4)
            test_layout.addWidget(stat_value, 3)
            test_layout.addWidget(pval_value, 3)
            diagnostic_grid.addLayout(test_layout)
            
            # Cache labels
            key_base = test_name.lower().replace('-', '_').replace(' ', '_')
            self._label_cache[f'{key_base}_stat'] = stat_value
            self._label_cache[f'{key_base}_pval'] = pval_value
        
        # Apply styling to all group boxes
        style_group_box(self._criteria_group)
        style_group_box(self._likelihood_group)
        style_group_box(self._diagnostic_group)
        
        # Add all groups to main layout
        self._main_layout.addWidget(self._criteria_group)
        self._main_layout.addWidget(self._likelihood_group)
        self._main_layout.addWidget(self._diagnostic_group)
        
        # Set layout on this widget
        self.setLayout(self._main_layout)
    
    def set_metrics_data(self, metrics_data: dict) -> None:
        """
        Updates the displayed metrics with new data.
        
        Args:
            metrics_data: Dictionary containing metric names and values
        """
        # Store metrics data
        self._metrics_data = metrics_data
        
        # Update displays
        self.update_information_criteria()
        self.update_likelihood_statistics()
        self.update_diagnostic_tests()
        
        # Apply styling to certain metrics
        if 'rsquared' in metrics_data:
            self.style_metric_value('rsquared_value', metrics_data['rsquared'])
        
        # Add additional styling for p-values
        for key, value in metrics_data.items():
            if '_pval' in key and key in self._label_cache:
                self.style_metric_value(key, value)
        
        logger.debug(f"Updated statistical metrics display with {len(metrics_data)} metrics")
    
    def clear(self) -> None:
        """
        Clears all metrics data and resets displays.
        """
        self._metrics_data = {}
        
        # Reset all value labels to "-"
        for label in self._label_cache.values():
            label.setText("-")
        
        logger.debug("Cleared statistical metrics display")
    
    def update_information_criteria(self) -> None:
        """
        Updates the display of information criteria metrics.
        """
        # Get criteria values from metrics data
        aic = self._metrics_data.get('aic')
        bic = self._metrics_data.get('bic')
        hqic = self._metrics_data.get('hqic')
        
        # Update labels
        self._label_cache['aic_value'].setText(format_metric_value(aic))
        self._label_cache['bic_value'].setText(format_metric_value(bic))
        self._label_cache['hqic_value'].setText(format_metric_value(hqic))
    
    def update_likelihood_statistics(self) -> None:
        """
        Updates the display of likelihood-based statistics.
        """
        # Get likelihood statistics from metrics data
        loglike = self._metrics_data.get('loglike') or self._metrics_data.get('log_likelihood')
        ssr = self._metrics_data.get('ssr') or self._metrics_data.get('sum_squared_residuals')
        rsquared = self._metrics_data.get('rsquared') or self._metrics_data.get('r_squared')
        
        # Update labels
        self._label_cache['loglike_value'].setText(format_metric_value(loglike))
        self._label_cache['ssr_value'].setText(format_metric_value(ssr))
        self._label_cache['rsquared_value'].setText(format_metric_value(rsquared))
    
    def update_diagnostic_tests(self) -> None:
        """
        Updates the display of diagnostic test results.
        """
        # Process Ljung-Box test
        ljung_box_stat = self._metrics_data.get('ljung_box_stat') 
        ljung_box_pval = self._metrics_data.get('ljung_box_pval')
        
        if ljung_box_stat is not None:
            self._label_cache['ljung_box_stat'].setText(format_metric_value(ljung_box_stat))
        
        if ljung_box_pval is not None:
            pval_label = self._label_cache['ljung_box_pval']
            pval_label.setText(format_metric_value(ljung_box_pval))
            self.style_metric_value('ljung_box_pval', ljung_box_pval)
        
        # Process Jarque-Bera test
        jb_stat = self._metrics_data.get('jarque_bera_stat') or self._metrics_data.get('jb_stat')
        jb_pval = self._metrics_data.get('jarque_bera_pval') or self._metrics_data.get('jb_pval')
        
        if jb_stat is not None:
            self._label_cache['jarque_bera_stat'].setText(format_metric_value(jb_stat))
        
        if jb_pval is not None:
            pval_label = self._label_cache['jarque_bera_pval']
            pval_label.setText(format_metric_value(jb_pval))
            self.style_metric_value('jarque_bera_pval', jb_pval)
        
        # Process Heteroskedasticity test
        het_stat = self._metrics_data.get('het_stat')
        het_pval = self._metrics_data.get('het_pval')
        
        if het_stat is not None:
            self._label_cache['heteroskedasticity_stat'].setText(format_metric_value(het_stat))
        
        if het_pval is not None:
            pval_label = self._label_cache['heteroskedasticity_pval']
            pval_label.setText(format_metric_value(het_pval))
            self.style_metric_value('heteroskedasticity_pval', het_pval)
        
        # Process Serial Correlation test
        serial_stat = self._metrics_data.get('serial_corr_stat')
        serial_pval = self._metrics_data.get('serial_corr_pval')
        
        if serial_stat is not None:
            self._label_cache['serial_correlation_stat'].setText(format_metric_value(serial_stat))
        
        if serial_pval is not None:
            pval_label = self._label_cache['serial_correlation_pval']
            pval_label.setText(format_metric_value(serial_pval))
            self.style_metric_value('serial_correlation_pval', serial_pval)
    
    def style_metric_value(self, metric_key: str, value: Any) -> None:
        """
        Applies styling to a metric value based on its value.
        
        Args:
            metric_key: The key of the metric in the label_cache
            value: The value to evaluate for styling
        """
        if metric_key not in self._label_cache:
            return
        
        label = self._label_cache[metric_key]
        
        # Style based on value type and magnitude
        if isinstance(value, float):
            # Example: Color coding based on R-squared value
            if metric_key == 'rsquared_value':
                if value < 0.3:
                    label.setStyleSheet("color: red;")  # Poor fit
                elif value < 0.7:
                    label.setStyleSheet("color: orange;")  # Moderate fit
                else:
                    label.setStyleSheet("color: green;")  # Good fit
        
        # Style for significance tests with p-values
        elif '_pval' in metric_key:
            try:
                p_value = float(value)
                if p_value < 0.01:
                    label.setStyleSheet("color: darkred; font-weight: bold;")  # Highly significant
                elif p_value < 0.05:
                    label.setStyleSheet("color: red;")  # Significant
                elif p_value < 0.1:
                    label.setStyleSheet("color: orange;")  # Marginally significant
                else:
                    label.setStyleSheet("")  # Not significant
            except (ValueError, TypeError):
                label.setStyleSheet("")
    
    def add_custom_metric(self, name: str, value: Any, group: str) -> None:
        """
        Adds a custom metric not in the standard categories.
        
        Args:
            name: The name of the metric
            value: The value of the metric
            group: The group to add the metric to ('criteria', 'likelihood', or 'diagnostic')
        """
        # Determine target group box
        target_layout = None
        if group.lower() == 'criteria':
            # Get the first column layout from the criteria group
            target_layout = self._criteria_group.layout().itemAt(0).layout()
            target_value_layout = self._criteria_group.layout().itemAt(1).layout()
        elif group.lower() == 'likelihood':
            # Get the first column layout from the likelihood group
            target_layout = self._likelihood_group.layout().itemAt(0).layout()
            target_value_layout = self._likelihood_group.layout().itemAt(1).layout()
        elif group.lower() == 'diagnostic':
            # Create a new row in the diagnostic tests group
            target_layout = QHBoxLayout()
            self._diagnostic_group.layout().addLayout(target_layout)
            
            name_label = QLabel(name)
            name_label.setFont(get_small_font())
            
            value_label = QLabel(format_metric_value(value))
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_label.setFont(get_small_font())
            
            # Special case for diagnostic group - different layout
            target_layout.addWidget(name_label, 4)
            target_layout.addWidget(value_label, 3)
            target_layout.addWidget(QLabel(""), 3)  # Empty label for alignment
            
            # Cache the value label
            cache_key = f"custom_{name.lower().replace(' ', '_')}"
            self._label_cache[cache_key] = value_label
            
            logger.debug(f"Added custom diagnostic metric: {name}")
            return
        else:
            logger.warning(f"Unknown metric group: {group}")
            return
        
        # Create labels for the new metric
        name_label, value_label = create_metric_label(name, value, False)
        
        # Add to layouts
        target_layout.addWidget(name_label)
        target_value_layout.addWidget(value_label)
        
        # Cache the value label
        cache_key = f"custom_{name.lower().replace(' ', '_')}"
        self._label_cache[cache_key] = value_label
        
        logger.debug(f"Added custom metric '{name}' to group '{group}'")
    
    def get_metrics_data(self) -> dict:
        """
        Returns the current metrics data dictionary.
        
        Returns:
            Current metrics data or empty dict if none
        """
        return self._metrics_data.copy()
    
    def connect_signals(self, signals: ModelSignals) -> None:
        """
        Connects this widget to model signals for automatic updates.
        
        Args:
            signals: ModelSignals instance to connect to
        """
        if signals:
            self.model_signals = signals
            signals.metrics_updated.connect(self._handle_metrics_updated)
            logger.debug("Connected to model signals")
    
    @pyqtSlot(object)
    def _handle_metrics_updated(self, metrics):
        """
        Slot method that handles the metrics_updated signal.
        
        Args:
            metrics: Metrics data object or dictionary
        """
        # Convert to dict if not already
        if not isinstance(metrics, dict):
            try:
                metrics = dict(metrics)
            except (ValueError, TypeError):
                logger.error("Received metrics is not convertible to dictionary")
                return
        
        self.set_metrics_data(metrics)
        logger.debug("Updated metrics from signal")