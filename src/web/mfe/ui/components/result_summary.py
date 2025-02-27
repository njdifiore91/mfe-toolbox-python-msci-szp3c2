#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt6-based UI component for displaying comprehensive model estimation result summaries.

This module provides the ResultSummary widget which integrates parameter tables, 
statistical metrics, diagnostic information, and model equations into a single
comprehensive display for econometric model results.
"""

from typing import Dict, List, Optional, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, 
    QTabWidget, QPushButton, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize

# Import internal components
from .parameter_table import ParameterTable
from .statistical_metrics import StatisticalMetrics
from .model_equation import ModelEquation
from .navigation_bar import NavigationBar

import logging

# Module logger
logger = logging.getLogger(__name__)


class ResultSummary(QWidget):
    """
    A PyQt6 widget that provides a comprehensive summary of model estimation results,
    including parameter tables, statistical metrics, and model equations.
    
    This widget integrates multiple specialized components to present a complete
    view of econometric model estimation results with proper formatting and organization.
    """
    
    # Signal emitted when results are updated
    resultsUpdated = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the ResultSummary widget with layout and child components.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Initialize internal variables
        self.parameter_data = []
        self.statistical_metrics = {}
        self.model_equation = ""
        self.model_info = {}
        self.current_page = 1
        self.total_pages = 1
        
        # Create components
        self._create_layout()
        
        # Initialize empty state
        self.clear()
        
        logger.debug("ResultSummary widget initialized")
    
    def _create_layout(self) -> None:
        """
        Internal method to create and set up the widget layout.
        """
        # Main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Title and model info section
        self.title_label = QLabel("Model Estimation Results")
        title_font = self.title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(14)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.model_info_group = QGroupBox("Model Information")
        model_info_layout = QVBoxLayout()
        self.model_type_label = QLabel()
        self.model_spec_label = QLabel()
        self.sample_info_label = QLabel()
        
        model_info_layout.addWidget(self.model_type_label)
        model_info_layout.addWidget(self.model_spec_label)
        model_info_layout.addWidget(self.sample_info_label)
        self.model_info_group.setLayout(model_info_layout)
        
        # Model equation section
        self.equation_group = QGroupBox("Model Equation")
        equation_layout = QVBoxLayout()
        self.equation_widget = ModelEquation()
        equation_layout.addWidget(self.equation_widget)
        self.equation_group.setLayout(equation_layout)
        
        # Parameter table section
        self.parameter_group = QGroupBox("Parameter Estimates")
        parameter_layout = QVBoxLayout()
        self.parameter_table = ParameterTable()
        parameter_layout.addWidget(self.parameter_table)
        self.parameter_group.setLayout(parameter_layout)
        
        # Statistical metrics section
        self.metrics_group = QGroupBox("Statistical Metrics")
        metrics_layout = QVBoxLayout()
        self.metrics_widget = StatisticalMetrics()
        metrics_layout.addWidget(self.metrics_widget)
        self.metrics_group.setLayout(metrics_layout)
        
        # Navigation bar (for multi-page results)
        self.navigation_layout = QHBoxLayout()
        self.navigation_bar = NavigationBar()
        self.navigation_bar.navigated_previous.connect(
            lambda: self._on_page_changed(self.current_page - 1)
        )
        self.navigation_bar.navigated_next.connect(
            lambda: self._on_page_changed(self.current_page + 1)
        )
        self.navigation_layout.addStretch()
        self.navigation_layout.addWidget(self.navigation_bar)
        self.navigation_layout.addStretch()
        
        # Tab widget for detailed results
        self.tab_widget = QTabWidget()
        
        # Summary tab - contains all the summary information
        summary_tab = QWidget()
        summary_layout = QVBoxLayout()
        summary_layout.addWidget(self.model_info_group)
        summary_layout.addWidget(self.equation_group)
        summary_layout.addWidget(self.parameter_group)
        summary_layout.addWidget(self.metrics_group)
        summary_layout.addLayout(self.navigation_layout)
        summary_tab.setLayout(summary_layout)
        
        self.tab_widget.addTab(summary_tab, "Summary")
        
        # Create export buttons
        button_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.clear_button = QPushButton("Clear Results")
        self.clear_button.clicked.connect(self.clear)
        
        button_layout.addStretch()
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.clear_button)
        
        # Add all components to main layout
        self.main_layout.addWidget(self.title_label)
        self.main_layout.addWidget(self.tab_widget)
        self.main_layout.addLayout(button_layout)
        
        # Set the layout
        self.setLayout(self.main_layout)
    
    def update_results(self, model_results: Dict[str, Any]) -> None:
        """
        Update the result summary with new model estimation results.
        
        Args:
            model_results: Dictionary containing model estimation results
        """
        try:
            # Extract relevant data from model results
            self.model_info = model_results.get('model_info', {})
            self.parameter_data = model_results.get('parameters', [])
            self.statistical_metrics = model_results.get('metrics', {})
            self.model_equation = model_results.get('equation', '')
            
            # Update model information display
            model_type = self.model_info.get('type', 'Unknown Model')
            self.title_label.setText(f"{model_type} Estimation Results")
            
            self.model_type_label.setText(f"Model Type: {model_type}")
            
            spec_info = self.model_info.get('specification', '')
            if spec_info:
                self.model_spec_label.setText(f"Specification: {spec_info}")
                self.model_spec_label.setVisible(True)
            else:
                self.model_spec_label.setVisible(False)
            
            sample_info = self.model_info.get('sample_info', '')
            if sample_info:
                self.sample_info_label.setText(f"Sample: {sample_info}")
                self.sample_info_label.setVisible(True)
            else:
                self.sample_info_label.setVisible(False)
            
            # Update model equation
            if self.model_equation:
                self.equation_widget.set_custom_equation(self.model_equation)
                self.equation_group.setVisible(True)
            else:
                self.equation_group.setVisible(False)
            
            # Update parameter table
            if self.parameter_data:
                param_dict = {}
                for param in self.parameter_data:
                    param_name = param.get('name', f"Parameter {len(param_dict) + 1}")
                    param_dict[param_name] = {
                        'estimate': param.get('estimate'),
                        'std_error': param.get('std_error'),
                        't_stat': param.get('t_stat'),
                        'p_value': param.get('p_value')
                    }
                self.parameter_table.set_parameter_data(param_dict)
                self.parameter_group.setVisible(True)
            else:
                self.parameter_group.setVisible(False)
            
            # Update statistical metrics
            if self.statistical_metrics:
                self.metrics_widget.set_metrics_data(self.statistical_metrics)
                self.metrics_group.setVisible(True)
            else:
                self.metrics_group.setVisible(False)
            
            # Update navigation if multiple pages
            multi_page_results = model_results.get('pages', None)
            if multi_page_results:
                self.total_pages = len(multi_page_results)
                self.current_page = 1  # Reset to first page on new results
                self.navigation_bar.update_page_indicator(self.current_page, self.total_pages)
                self.navigation_layout.setVisible(True)
            else:
                self.navigation_layout.setVisible(False)
            
            # Add any additional tabs for detailed results
            detailed_results = model_results.get('detailed_results', {})
            # Clear any existing detail tabs (keep summary tab)
            while self.tab_widget.count() > 1:
                self.tab_widget.removeTab(1)
                
            # Add detailed result tabs
            for tab_name, tab_content in detailed_results.items():
                tab_widget = QWidget()
                tab_layout = QVBoxLayout()
                content_label = QLabel(str(tab_content))
                content_label.setWordWrap(True)
                tab_layout.addWidget(content_label)
                tab_widget.setLayout(tab_layout)
                self.tab_widget.addTab(tab_widget, tab_name)
            
            # Emit signal that results have been updated
            self.resultsUpdated.emit()
            
            logger.debug(f"Updated result summary with {model_type} results")
            
        except Exception as e:
            logger.error(f"Error updating results: {str(e)}")
            QMessageBox.critical(
                self, 
                "Error Updating Results", 
                f"An error occurred while updating results: {str(e)}"
            )
    
    def clear(self) -> None:
        """
        Reset the result summary to its initial empty state.
        """
        # Clear all data
        self.parameter_data = []
        self.statistical_metrics = {}
        self.model_equation = ""
        self.model_info = {}
        
        # Reset UI components
        self.title_label.setText("Model Estimation Results")
        self.model_type_label.setText("Model Type: None")
        self.model_spec_label.setText("")
        self.sample_info_label.setText("")
        
        # Clear sub-components
        self.equation_widget.clear()
        self.parameter_table.clear()
        self.metrics_widget.clear()
        
        # Hide groups that should be empty
        self.equation_group.setVisible(False)
        
        # Reset navigation
        self.current_page = 1
        self.total_pages = 1
        self.navigation_layout.setVisible(False)
        
        # Remove detail tabs (keep summary tab)
        while self.tab_widget.count() > 1:
            self.tab_widget.removeTab(1)
        
        logger.debug("Cleared result summary")
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save the current result summary to a file.
        
        Args:
            file_path: Path to save the results file
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Format the results as text
            result_text = self.format_as_text()
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(result_text)
            
            logger.info(f"Saved results to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results to file: {str(e)}")
            QMessageBox.critical(
                self, 
                "Save Failed", 
                f"Failed to save results to file: {str(e)}"
            )
            return False
    
    def format_as_text(self) -> str:
        """
        Format the current results as plain text for export or display.
        
        Returns:
            Formatted text representation of results
        """
        lines = []
        
        # Add title
        model_type = self.model_info.get('type', 'Unknown Model')
        lines.append(f"{model_type} Estimation Results")
        lines.append("=" * len(lines[0]))
        lines.append("")
        
        # Add model information
        lines.append("MODEL INFORMATION")
        lines.append("-----------------")
        spec_info = self.model_info.get('specification', '')
        if spec_info:
            lines.append(f"Specification: {spec_info}")
        
        sample_info = self.model_info.get('sample_info', '')
        if sample_info:
            lines.append(f"Sample: {sample_info}")
        
        lines.append("")
        
        # Add model equation (as plain text)
        if self.model_equation:
            lines.append("MODEL EQUATION")
            lines.append("-------------")
            # Strip LaTeX markers for plain text
            plain_eq = self.model_equation.replace('$', '').replace('\\', '')
            lines.append(plain_eq)
            lines.append("")
        
        # Add parameter table
        if self.parameter_data:
            lines.append("PARAMETER ESTIMATES")
            lines.append("------------------")
            lines.append(self._format_parameters())
            lines.append("")
        
        # Add statistical metrics
        if self.statistical_metrics:
            lines.append("STATISTICAL METRICS")
            lines.append("------------------")
            lines.append(self._format_metrics())
            lines.append("")
        
        return "\n".join(lines)
    
    def _on_page_changed(self, page_number: int) -> None:
        """
        Internal slot to handle page navigation events.
        
        Args:
            page_number: Page number to navigate to
        """
        if page_number < 1 or page_number > self.total_pages:
            return
        
        self.current_page = page_number
        self.navigation_bar.update_page_indicator(self.current_page, self.total_pages)
        
        logger.debug(f"Navigated to page {page_number} of {self.total_pages}")
        
        # TODO: Update content with page-specific data
        # In a real implementation, we would update the displayed content
        # based on the current page
    
    def _format_parameters(self) -> str:
        """
        Internal method to format parameter data for text output.
        
        Returns:
            Formatted parameter table as text
        """
        if not self.parameter_data:
            return "No parameter data available."
        
        lines = []
        
        # Format header
        param_col = "Parameter"
        est_col = "Estimate"
        std_col = "Std.Error"
        tstat_col = "t-stat"
        pval_col = "p-value"
        
        # Calculate column widths
        param_width = max(len(param_col), max(len(param.get('name', '')) for param in self.parameter_data))
        est_width = max(len(est_col), 10)  # Assuming typical width for numbers
        std_width = max(len(std_col), 10)
        tstat_width = max(len(tstat_col), 8)
        pval_width = max(len(pval_col), 8)
        
        # Format header row
        header = (
            f"{param_col:<{param_width}}  "
            f"{est_col:>{est_width}}  "
            f"{std_col:>{std_width}}  "
            f"{tstat_col:>{tstat_width}}  "
            f"{pval_col:>{pval_width}}"
        )
        lines.append(header)
        
        # Add separator line
        lines.append("-" * len(header))
        
        # Format each parameter row
        for param in self.parameter_data:
            name = param.get('name', '')
            estimate = param.get('estimate')
            std_error = param.get('std_error')
            t_stat = param.get('t_stat')
            p_value = param.get('p_value')
            
            # Format numbers
            est_str = f"{estimate:.6f}" if estimate is not None else "-"
            std_str = f"{std_error:.6f}" if std_error is not None else "-"
            tstat_str = f"{t_stat:.4f}" if t_stat is not None else "-"
            pval_str = f"{p_value:.6f}" if p_value is not None else "-"
            
            # Format full row
            row = (
                f"{name:<{param_width}}  "
                f"{est_str:>{est_width}}  "
                f"{std_str:>{std_width}}  "
                f"{tstat_str:>{tstat_width}}  "
                f"{pval_str:>{pval_width}}"
            )
            lines.append(row)
        
        return "\n".join(lines)
    
    def _format_metrics(self) -> str:
        """
        Internal method to format statistical metrics for text output.
        
        Returns:
            Formatted metrics as text
        """
        if not self.statistical_metrics:
            return "No statistical metrics available."
        
        lines = []
        
        # Format metrics in logical groups
        
        # Information criteria
        info_criteria = []
        if 'aic' in self.statistical_metrics:
            info_criteria.append(f"AIC: {self.statistical_metrics['aic']:.6f}")
        if 'bic' in self.statistical_metrics:
            info_criteria.append(f"BIC: {self.statistical_metrics['bic']:.6f}")
        if 'hqic' in self.statistical_metrics:
            info_criteria.append(f"HQIC: {self.statistical_metrics['hqic']:.6f}")
        
        if info_criteria:
            lines.append("Information Criteria:")
            lines.extend(f"  {criterion}" for criterion in info_criteria)
            lines.append("")
        
        # Likelihood statistics
        likelihood_stats = []
        if 'loglike' in self.statistical_metrics:
            likelihood_stats.append(f"Log-Likelihood: {self.statistical_metrics['loglike']:.6f}")
        if 'rsquared' in self.statistical_metrics:
            likelihood_stats.append(f"R-squared: {self.statistical_metrics['rsquared']:.6f}")
        if 'ssr' in self.statistical_metrics:
            likelihood_stats.append(f"Sum of Squared Residuals: {self.statistical_metrics['ssr']:.6f}")
        
        if likelihood_stats:
            lines.append("Likelihood Statistics:")
            lines.extend(f"  {stat}" for stat in likelihood_stats)
            lines.append("")
        
        # Diagnostic tests
        diagnostic_tests = []
        test_pairs = [
            ('ljung_box_stat', 'ljung_box_pval', 'Ljung-Box Test'),
            ('jarque_bera_stat', 'jarque_bera_pval', 'Jarque-Bera Test'),
            ('het_stat', 'het_pval', 'Heteroskedasticity Test'),
            ('serial_corr_stat', 'serial_corr_pval', 'Serial Correlation Test')
        ]
        
        for stat_key, pval_key, test_name in test_pairs:
            if stat_key in self.statistical_metrics and pval_key in self.statistical_metrics:
                stat = self.statistical_metrics[stat_key]
                pval = self.statistical_metrics[pval_key]
                diagnostic_tests.append(
                    f"{test_name}: Stat = {stat:.6f}, p-value = {pval:.6f}"
                )
        
        if diagnostic_tests:
            lines.append("Diagnostic Tests:")
            lines.extend(f"  {test}" for test in diagnostic_tests)
            lines.append("")
        
        # Other metrics (anything not caught in the specific sections)
        other_metrics = [
            f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in self.statistical_metrics.items()
            if (
                key not in {'aic', 'bic', 'hqic', 'loglike', 'rsquared', 'ssr'} and
                not key.endswith('_stat') and not key.endswith('_pval')
            )
        ]
        
        if other_metrics:
            lines.append("Other Metrics:")
            lines.extend(f"  {metric}" for metric in other_metrics)
        
        return "\n".join(lines)
    
    def _on_export_clicked(self) -> None:
        """
        Handles the export button click event.
        """
        if not self.model_info:
            QMessageBox.warning(
                self, 
                "No Results", 
                "There are no results to export."
            )
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            success = self.save_to_file(file_path)
            if success:
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Results exported to {file_path}"
                )
    
    def sizeHint(self) -> QSize:
        """
        Returns a recommended size for this widget.
        
        Returns:
            QSize: Recommended size
        """
        return QSize(800, 600)