#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARMAX Model Results Viewer for MFE Toolbox.

This module implements a PyQt6-based dialog for displaying ARMAX model estimation
results with interactive navigation, parameter tables, model equations, and 
diagnostic plots as specified in the UI design.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Union

import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
    QPushButton, QGroupBox, QTabWidget, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QPixmap

from ..components.model_equation import ModelEquation
from ..components.parameter_table import ParameterTable
from ..components.statistical_metrics import StatisticalMetrics
from ..components.navigation_bar import NavigationBar
from ..plots.residual_plot import ResidualPlot

# Configure module logger
logger = logging.getLogger(__name__)


class ARMAXViewer(QDialog):
    """
    A dialog for displaying ARMAX model estimation results with interactive
    navigation, parameter tables, model equations, and diagnostic plots.
    
    This dialog presents estimation results from ARMAX models, providing a
    comprehensive view of parameter estimates, statistical metrics, and
    diagnostic visualizations. It supports navigating between multiple model
    results for comparison.
    """
    
    def __init__(self, model_results: List, parent: Optional[QWidget] = None):
        """
        Initialize the ARMAX Results Viewer dialog.
        
        Parameters
        ----------
        model_results : List
            List of model estimation result objects to display
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Store model results
        self._results = model_results
        self._current_page = 0
        self._total_pages = len(model_results) if model_results else 0
        
        # Set window properties
        self.setWindowTitle("ARMAX Model Results")
        self.setMinimumSize(800, 600)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Create and add the equation section
        equation_section = self.create_equation_section()
        main_layout.addWidget(equation_section)
        
        # Create and add the parameter section
        parameter_section = self.create_parameter_section()
        main_layout.addWidget(parameter_section)
        
        # Create and add the statistics section
        statistics_section = self.create_statistics_section()
        main_layout.addWidget(statistics_section)
        
        # Create and add the plots section
        plots_section = self.create_plots_section()
        main_layout.addWidget(plots_section)
        
        # Create and add the navigation section
        navigation_section = self.create_navigation_section()
        main_layout.addWidget(navigation_section)
        
        # Connect navigation signals to slots
        self._navigation_bar.navigated_previous.connect(self.on_previous)
        self._navigation_bar.navigated_next.connect(self.on_next)
        
        # Set layout on dialog
        self.setLayout(main_layout)
        
        # Initialize display with first result if available
        if self._total_pages > 0:
            self.update_display()
        
        logger.debug("ARMAXViewer initialized with %d model results", self._total_pages)
    
    def create_equation_section(self) -> QWidget:
        """
        Creates the model equation section of the dialog.
        
        Returns
        -------
        QWidget
            The equation section widget
        """
        # Create group box with title
        group_box = QGroupBox("Model Equation")
        layout = QVBoxLayout()
        
        # Create equation widget
        self._equation_widget = ModelEquation()
        
        # Add to layout
        layout.addWidget(self._equation_widget)
        
        # Set layout on group box
        group_box.setLayout(layout)
        
        return group_box
    
    def create_parameter_section(self) -> QWidget:
        """
        Creates the parameter estimates section of the dialog.
        
        Returns
        -------
        QWidget
            The parameter section widget
        """
        # Create group box with title
        group_box = QGroupBox("Parameter Estimates")
        layout = QVBoxLayout()
        
        # Create parameter table
        self._parameter_table = ParameterTable()
        
        # Add to layout
        layout.addWidget(self._parameter_table)
        
        # Set layout on group box
        group_box.setLayout(layout)
        
        return group_box
    
    def create_statistics_section(self) -> QWidget:
        """
        Creates the statistical metrics section of the dialog.
        
        Returns
        -------
        QWidget
            The statistics section widget
        """
        # Create group box with title
        group_box = QGroupBox("Statistical Metrics")
        layout = QVBoxLayout()
        
        # Create statistics widget
        self._statistics_widget = StatisticalMetrics()
        
        # Add to layout
        layout.addWidget(self._statistics_widget)
        
        # Set layout on group box
        group_box.setLayout(layout)
        
        return group_box
    
    def create_plots_section(self) -> QWidget:
        """
        Creates the diagnostic plots section of the dialog.
        
        Returns
        -------
        QWidget
            The plots section widget
        """
        # Create group box with title
        group_box = QGroupBox("Diagnostic Plots")
        layout = QVBoxLayout()
        
        # Create tab widget for plots
        self._plot_tabs = QTabWidget()
        
        # Create residual plot
        self._residual_plot = ResidualPlot()
        
        # Add plot to tabs
        self._plot_tabs.addTab(self._residual_plot, "Residuals")
        
        # Add tabs to layout
        layout.addWidget(self._plot_tabs)
        
        # Set layout on group box
        group_box.setLayout(layout)
        
        return group_box
    
    def create_navigation_section(self) -> QWidget:
        """
        Creates the navigation controls section of the dialog.
        
        Returns
        -------
        QWidget
            The navigation section widget
        """
        # Create widget for navigation controls
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 5, 0, 0)
        
        # Create navigation bar
        self._navigation_bar = NavigationBar()
        
        # Create close button
        self._close_button = QPushButton("Close")
        self._close_button.clicked.connect(self.close)
        
        # Add widgets to layout
        layout.addWidget(self._navigation_bar, 1)  # Stretch factor of 1
        layout.addWidget(self._close_button)
        
        # Set layout on widget
        widget.setLayout(layout)
        
        return widget
    
    def update_display(self) -> None:
        """
        Updates all components with current model results.
        
        This method updates the display with data from the current result,
        including model equation, parameter estimates, statistics, and plots.
        """
        if self._total_pages == 0:
            logger.warning("No model results to display")
            return
        
        # Get current result
        result = self._results[self._current_page]
        
        try:
            # Extract model parameters
            ar_order = getattr(result, 'ar_order', 0)
            ma_order = getattr(result, 'ma_order', 0)
            include_constant = getattr(result, 'include_constant', False)
            
            # Extract parameter values
            params = {}
            if hasattr(result, 'params'):
                for name, value in zip(getattr(result, 'param_names', []), result.params):
                    params[name] = value
            
            # Update equation widget
            self._equation_widget.set_arma_equation(ar_order, ma_order, include_constant, params)
            
            # Update parameter table
            param_data = {}
            if hasattr(result, 'params'):
                for i, name in enumerate(getattr(result, 'param_names', [])):
                    param_data[name] = {
                        'estimate': result.params[i],
                        'std_error': result.bse[i] if hasattr(result, 'bse') else None,
                        't_stat': result.tvalues[i] if hasattr(result, 'tvalues') else None,
                        'p_value': result.pvalues[i] if hasattr(result, 'pvalues') else None
                    }
            self._parameter_table.set_parameter_data(param_data)
            
            # Update statistics widget
            metrics = {
                'loglike': getattr(result, 'loglike', None),
                'aic': getattr(result, 'aic', None),
                'bic': getattr(result, 'bic', None),
                'hqic': getattr(result, 'hqic', None),
                'rsquared': getattr(result, 'rsquared', None),
                'ssr': getattr(result, 'ssr', None)
            }
            
            # Add any diagnostic test results if available
            if hasattr(result, 'diagnostic_tests'):
                for test_name, test_result in result.diagnostic_tests.items():
                    if isinstance(test_result, tuple) and len(test_result) >= 2:
                        metrics[f'{test_name}_stat'] = test_result[0]
                        metrics[f'{test_name}_pval'] = test_result[1]
            
            self._statistics_widget.set_metrics_data(metrics)
            
            # Update residual plot
            if hasattr(result, 'resid'):
                self._residual_plot.set_residuals(result.resid)
            
            if hasattr(result, 'fittedvalues'):
                self._residual_plot.set_fitted_values(result.fittedvalues)
            
            self._residual_plot.update_plot()
            
            # Update navigation bar
            self._navigation_bar.update_page_indicator(self._current_page + 1, self._total_pages)
            
            logger.debug("Updated display with result %d/%d", 
                        self._current_page + 1, self._total_pages)
            
        except Exception as e:
            logger.error("Error updating display: %s", str(e))
    
    async def update_display_async(self) -> None:
        """
        Asynchronously updates display to prevent UI blocking.
        
        This method performs the display update asynchronously, allowing the UI
        to remain responsive during computationally intensive plot updates.
        """
        if self._total_pages == 0:
            logger.warning("No model results to display")
            return
        
        # Get current result
        result = self._results[self._current_page]
        
        try:
            # Extract model parameters
            ar_order = getattr(result, 'ar_order', 0)
            ma_order = getattr(result, 'ma_order', 0)
            include_constant = getattr(result, 'include_constant', False)
            
            # Extract parameter values
            params = {}
            if hasattr(result, 'params'):
                for name, value in zip(getattr(result, 'param_names', []), result.params):
                    params[name] = value
            
            # Update equation widget
            self._equation_widget.set_arma_equation(ar_order, ma_order, include_constant, params)
            
            # Update parameter table
            param_data = {}
            if hasattr(result, 'params'):
                for i, name in enumerate(getattr(result, 'param_names', [])):
                    param_data[name] = {
                        'estimate': result.params[i],
                        'std_error': result.bse[i] if hasattr(result, 'bse') else None,
                        't_stat': result.tvalues[i] if hasattr(result, 'tvalues') else None,
                        'p_value': result.pvalues[i] if hasattr(result, 'pvalues') else None
                    }
            self._parameter_table.set_parameter_data(param_data)
            
            # Update statistics widget
            metrics = {
                'loglike': getattr(result, 'loglike', None),
                'aic': getattr(result, 'aic', None),
                'bic': getattr(result, 'bic', None),
                'hqic': getattr(result, 'hqic', None),
                'rsquared': getattr(result, 'rsquared', None),
                'ssr': getattr(result, 'ssr', None)
            }
            
            # Add any diagnostic test results if available
            if hasattr(result, 'diagnostic_tests'):
                for test_name, test_result in result.diagnostic_tests.items():
                    if isinstance(test_result, tuple) and len(test_result) >= 2:
                        metrics[f'{test_name}_stat'] = test_result[0]
                        metrics[f'{test_name}_pval'] = test_result[1]
            
            self._statistics_widget.set_metrics_data(metrics)
            
            # Asynchronously update residual plot
            if hasattr(result, 'resid'):
                self._residual_plot.set_residuals(result.resid)
            
            if hasattr(result, 'fittedvalues'):
                self._residual_plot.set_fitted_values(result.fittedvalues)
            
            # Use the async update method for plot rendering
            await self._residual_plot.async_update_plot()
            
            # Update navigation bar
            self._navigation_bar.update_page_indicator(self._current_page + 1, self._total_pages)
            
            logger.debug("Asynchronously updated display with result %d/%d", 
                        self._current_page + 1, self._total_pages)
            
        except Exception as e:
            logger.error("Error in async display update: %s", str(e))
    
    def on_previous(self) -> None:
        """
        Handle previous page navigation.
        
        Moves to the previous result in the list and updates the display.
        """
        if self._current_page > 0:
            self._current_page -= 1
            self.update_display()
            logger.debug("Navigated to previous result: %d/%d", 
                        self._current_page + 1, self._total_pages)
    
    def on_next(self) -> None:
        """
        Handle next page navigation.
        
        Moves to the next result in the list and updates the display.
        """
        if self._current_page < self._total_pages - 1:
            self._current_page += 1
            self.update_display()
            logger.debug("Navigated to next result: %d/%d", 
                        self._current_page + 1, self._total_pages)
    
    def closeEvent(self, event) -> None:
        """
        Handle the dialog close event.
        
        Performs cleanup and logs the closure.
        
        Parameters
        ----------
        event : QCloseEvent
            The close event
        """
        logger.debug("Closing ARMAXViewer")
        
        # Clean up resources
        self._residual_plot.clear()
        
        # Accept the event to allow closing
        event.accept()