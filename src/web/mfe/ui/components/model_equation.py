#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Equation Widget for MFE Toolbox UI.

This module provides a PyQt6 widget for rendering and displaying econometric
model equations in LaTeX format, supporting ARMA, GARCH, and custom equations
with proper mathematical notation.
"""

from typing import Dict, Optional, Union, List, Any
import asyncio
import logging

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap

from ..latex_renderer import LatexRenderer

# Configure logger for this module
logger = logging.getLogger(__name__)

# Default widget settings
DEFAULT_FONT_SIZE = 12
DEFAULT_WIDTH = 600
DEFAULT_HEIGHT = 120

def _format_arma_equation(p: int, q: int, include_constant: bool, 
                          parameters: Optional[Dict[str, float]] = None) -> str:
    """
    Formats an ARMA model equation in LaTeX format with optional parameter values.
    
    Args:
        p: AR order
        q: MA order
        include_constant: Whether to include a constant term
        parameters: Optional dictionary of parameter values
        
    Returns:
        Formatted LaTeX equation for the ARMA model
    """
    # Initialize equation components
    ar_component = ""
    ma_component = ""
    constant_component = ""
    
    # Format AR component
    if p > 0:
        ar_terms = []
        for i in range(1, p + 1):
            if parameters and f"phi_{i}" in parameters:
                phi_value = parameters[f"phi_{i}"]
                phi_sign = "-" if phi_value < 0 else ""
                ar_terms.append(f"{phi_sign} {abs(phi_value):.3f} y_{{t-{i}}}")
            else:
                ar_terms.append(f"\\phi_{{{i}}} y_{{t-{i}}}")
        ar_component = " - ".join(ar_terms)
    
    # Format MA component
    if q > 0:
        ma_terms = []
        for i in range(1, q + 1):
            if parameters and f"theta_{i}" in parameters:
                theta_value = parameters[f"theta_{i}"]
                theta_sign = "+" if theta_value > 0 else ""
                ma_terms.append(f"{theta_sign} {abs(theta_value):.3f} \\varepsilon_{{t-{i}}}")
            else:
                ma_terms.append(f"\\theta_{{{i}}} \\varepsilon_{{t-{i}}}")
        ma_component = " + ".join(ma_terms)
    
    # Format constant term
    if include_constant:
        if parameters and "constant" in parameters:
            const_value = parameters["constant"]
            constant_component = f"{const_value:.4f}"
        else:
            constant_component = "\\mu"
    
    # Build the complete equation
    equation_parts = []
    
    # Start with y_t
    equation_parts.append("y_t")
    
    # Add constant if present
    if constant_component:
        equation_parts.append(f"= {constant_component}")
    else:
        equation_parts.append("=")
    
    # Add AR component if present
    if ar_component:
        if equation_parts[-1] != "=":
            equation_parts.append(f"+ {ar_component}")
        else:
            equation_parts.append(ar_component)
    
    # Add MA component if present
    if ma_component:
        equation_parts.append(f"+ {ma_component}")
    
    # Add error term
    equation_parts.append("+ \\varepsilon_t")
    
    # Join all parts
    equation = " ".join(equation_parts)
    
    # Wrap in LaTeX display math
    return f"${equation}$"

def _format_garch_equation(p: int, q: int, 
                           parameters: Optional[Dict[str, float]] = None) -> str:
    """
    Formats a GARCH model equation in LaTeX format with optional parameter values.
    
    Args:
        p: GARCH order
        q: ARCH order
        parameters: Optional dictionary of parameter values
        
    Returns:
        Formatted LaTeX equation for the GARCH model
    """
    # Initialize equation components
    arch_component = ""
    garch_component = ""
    
    # Format ARCH component (alpha terms)
    if q > 0:
        arch_terms = []
        for i in range(1, q + 1):
            if parameters and f"alpha_{i}" in parameters:
                alpha_value = parameters[f"alpha_{i}"]
                arch_terms.append(f"{alpha_value:.3f} \\varepsilon^2_{{t-{i}}}")
            else:
                arch_terms.append(f"\\alpha_{{{i}}} \\varepsilon^2_{{t-{i}}}")
        arch_component = " + ".join(arch_terms)
    
    # Format GARCH component (beta terms)
    if p > 0:
        garch_terms = []
        for i in range(1, p + 1):
            if parameters and f"beta_{i}" in parameters:
                beta_value = parameters[f"beta_{i}"]
                garch_terms.append(f"{beta_value:.3f} \\sigma^2_{{t-{i}}}")
            else:
                garch_terms.append(f"\\beta_{{{i}}} \\sigma^2_{{t-{i}}}")
        garch_component = " + ".join(garch_terms)
    
    # Format constant term (omega)
    if parameters and "omega" in parameters:
        omega_value = parameters["omega"]
        omega_component = f"{omega_value:.4f}"
    else:
        omega_component = "\\omega"
    
    # Build the complete equation
    equation = f"\\sigma^2_t = {omega_component}"
    
    # Add ARCH component if present
    if arch_component:
        equation += f" + {arch_component}"
    
    # Add GARCH component if present
    if garch_component:
        equation += f" + {garch_component}"
    
    # Wrap in LaTeX display math
    return f"${equation}$"

def _format_custom_equation(equation: str) -> str:
    """
    Prepares a custom LaTeX equation for rendering, ensuring proper LaTeX delimiters.
    
    Args:
        equation: Custom equation string
        
    Returns:
        Properly formatted LaTeX equation
    """
    # Check if equation already has LaTeX delimiters
    if equation.startswith('$') and equation.endswith('$'):
        return equation
    if equation.startswith('$$') and equation.endswith('$$'):
        return equation
    if equation.startswith('\\begin{equation}') and equation.endswith('\\end{equation}'):
        return equation
    
    # Add delimiters
    return f"${equation}$"

class ModelEquation(QWidget):
    """
    PyQt6 widget for displaying econometric model equations in LaTeX format with
    support for ARMA, GARCH, and custom equations.
    
    This widget renders mathematical equations for econometric models using
    LaTeX notation, providing specialized formatting for common model types
    and the ability to update parameter values dynamically.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the ModelEquation widget.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Initialize the renderer
        self._renderer = LatexRenderer()
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create label for displaying equation
        self._equation_label = QLabel()
        self._equation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._equation_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        # Initialize properties
        self.latex_text = ""
        self.font_size = DEFAULT_FONT_SIZE
        self.current_pixmap = None
        self._render_task = None
        
        # Add label to layout
        layout.addWidget(self._equation_label)
        
        # Set layout
        self.setLayout(layout)
        
        # Set minimum size
        self.setMinimumSize(DEFAULT_WIDTH, DEFAULT_HEIGHT)
        
        logger.debug("ModelEquation widget initialized")
    
    def set_arma_equation(self, p: int, q: int, include_constant: bool, 
                          parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Sets the widget to display an ARMA model equation.
        
        Args:
            p: AR order
            q: MA order
            include_constant: Whether to include a constant term
            parameters: Optional dictionary of parameter values
        """
        # Generate LaTeX equation
        latex_eq = _format_arma_equation(p, q, include_constant, parameters)
        
        # Store the LaTeX text
        self.latex_text = latex_eq
        
        # Render the equation
        self._render_equation()
        
        logger.debug(f"Set ARMA equation: p={p}, q={q}, constant={include_constant}")
    
    def set_garch_equation(self, p: int, q: int, 
                           parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Sets the widget to display a GARCH model equation.
        
        Args:
            p: GARCH order
            q: ARCH order
            parameters: Optional dictionary of parameter values
        """
        # Generate LaTeX equation
        latex_eq = _format_garch_equation(p, q, parameters)
        
        # Store the LaTeX text
        self.latex_text = latex_eq
        
        # Render the equation
        self._render_equation()
        
        logger.debug(f"Set GARCH equation: p={p}, q={q}")
    
    def set_custom_equation(self, equation: str) -> None:
        """
        Sets the widget to display a custom LaTeX equation.
        
        Args:
            equation: Custom equation string
        """
        # Format the equation
        latex_eq = _format_custom_equation(equation)
        
        # Store the LaTeX text
        self.latex_text = latex_eq
        
        # Render the equation
        self._render_equation()
        
        logger.debug("Set custom equation")
    
    def update_parameters(self, parameters: Dict[str, float]) -> None:
        """
        Updates the parameter values in the current equation.
        
        Args:
            parameters: Dictionary of parameter values
        """
        # Check the current equation type and re-render with updated parameters
        if "ARMA" in self.latex_text or "y_t" in self.latex_text:
            # Extract p, q, and include_constant from existing equation
            p = self.latex_text.count("phi")
            q = self.latex_text.count("theta")
            include_constant = "mu" in self.latex_text or any(c.isdigit() for c in self.latex_text.split("=")[1].strip())
            
            # Re-render with updated parameters
            self.set_arma_equation(p, q, include_constant, parameters)
        elif "sigma" in self.latex_text:
            # Extract p and q from existing equation
            p = self.latex_text.count("beta")
            q = self.latex_text.count("alpha")
            
            # Re-render with updated parameters
            self.set_garch_equation(p, q, parameters)
        else:
            # For custom equations, just re-render the existing equation
            # This doesn't update parameters, but keeps consistent behavior
            self._render_equation()
        
        logger.debug(f"Updated equation parameters: {', '.join(parameters.keys())}")
    
    def set_font_size(self, size: float) -> None:
        """
        Sets the font size for the displayed equation.
        
        Args:
            size: Font size in points
        """
        # Update font size
        self.font_size = size
        
        # Re-render the equation
        self._render_equation()
        
        logger.debug(f"Set equation font size: {size}")
    
    def clear(self) -> None:
        """
        Clears the current equation display.
        """
        # Cancel any pending render task
        if self._render_task is not None:
            self._render_task.cancel()
            self._render_task = None
        
        # Clear the stored text
        self.latex_text = ""
        
        # Clear the display
        self._equation_label.setPixmap(QPixmap())
        self.current_pixmap = None
        
        logger.debug("Cleared equation display")
    
    def _render_equation(self) -> None:
        """
        Renders the current LaTeX equation and updates the display.
        """
        # Check if there's any text to render
        if not self.latex_text:
            self.clear()
            return
        
        # Cancel any existing render task
        if self._render_task is not None:
            self._render_task.cancel()
            self._render_task = None
        
        # Prepare render options
        options = {
            'fontsize': self.font_size,
        }
        
        try:
            # Render the equation
            pixmap = self._renderer.render_to_pixmap(self.latex_text, options)
            
            # Update the display
            self._equation_label.setPixmap(pixmap)
            self.current_pixmap = pixmap
            
            logger.debug("Rendered equation")
        except Exception as e:
            logger.error(f"Error rendering equation: {str(e)}")
            # Clear display on error
            self._equation_label.setPixmap(QPixmap())
            self.current_pixmap = None
    
    async def _render_equation_async(self) -> None:
        """
        Asynchronously renders the equation to avoid blocking the UI.
        """
        # Check if there's any text to render
        if not self.latex_text:
            self.clear()
            return
        
        # Prepare render options
        options = {
            'fontsize': self.font_size,
        }
        
        try:
            # Render the equation asynchronously
            pixmap = await self._renderer.render_async(self.latex_text, options)
            
            # Update the display
            self._equation_label.setPixmap(pixmap)
            self.current_pixmap = pixmap
            
            logger.debug("Rendered equation asynchronously")
        except Exception as e:
            logger.error(f"Error in asynchronous equation rendering: {str(e)}")
            # Clear display on error
            self._equation_label.setPixmap(QPixmap())
            self.current_pixmap = None
    
    def get_latex_text(self) -> str:
        """
        Gets the current LaTeX equation text.
        
        Returns:
            Current LaTeX equation text
        """
        return self.latex_text
    
    def sizeHint(self) -> QSize:
        """
        Returns the recommended size for the widget.
        
        Returns:
            Recommended widget size
        """
        return QSize(DEFAULT_WIDTH, DEFAULT_HEIGHT)