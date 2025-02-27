#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for the ModelEquation UI component.

This module contains tests for the ModelEquation component which renders
LaTeX mathematical equations for financial econometric models in the
PyQt6-based interface.
"""

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from mfe.ui.components.model_equation import ModelEquation

# Create QApplication instance for tests
@pytest.fixture(scope="module")
def app():
    """Create a QApplication instance for the tests."""
    # Only create the app if it doesn't already exist
    if not QApplication.instance():
        app = QApplication([])
        yield app
        app.quit()
    else:
        yield QApplication.instance()

@pytest.fixture
def model_equation(app):
    """Create a ModelEquation instance for testing."""
    widget = ModelEquation()
    yield widget
    # Clean up after test
    widget.clear()
    widget.deleteLater()

def test_model_equation_init(model_equation):
    """Test the proper initialization of the ModelEquation component."""
    # Verify widget is properly initialized
    assert model_equation is not None
    # Check default properties are set correctly
    assert model_equation.latex_text == ""
    assert model_equation.font_size == 12  # Default font size from the implementation
    # Check that the label exists
    assert model_equation._equation_label is not None
    # Verify renderer is initialized
    assert model_equation._renderer is not None

def test_model_equation_set_equation(model_equation):
    """Test setting various equation types to the ModelEquation component."""
    # Test setting ARMA equation
    model_equation.set_arma_equation(1, 1, True)
    assert "y_t" in model_equation.latex_text
    assert "\\phi" in model_equation.latex_text
    assert "\\theta" in model_equation.latex_text
    assert "\\mu" in model_equation.latex_text
    
    # Test setting GARCH equation
    model_equation.set_garch_equation(1, 1)
    assert "\\sigma^2_t" in model_equation.latex_text
    assert "\\alpha" in model_equation.latex_text
    assert "\\beta" in model_equation.latex_text
    
    # Test setting custom equation
    test_equation = "y_t = \\alpha + \\beta x_t + \\varepsilon_t"
    model_equation.set_custom_equation(test_equation)
    assert "y_t" in model_equation.latex_text
    assert "\\alpha" in model_equation.latex_text
    assert "\\beta" in model_equation.latex_text

def test_model_equation_get_equation(model_equation):
    """Test retrieving the current equation from the ModelEquation component."""
    # Set a specific equation
    test_equation = "y_t = \\alpha + \\beta x_t + \\varepsilon_t"
    model_equation.set_custom_equation(test_equation)
    
    # Retrieve and verify the equation
    latex_text = model_equation.get_latex_text()
    assert latex_text is not None
    assert "$y_t = \\alpha + \\beta x_t + \\varepsilon_t$" in latex_text

def test_model_equation_update(model_equation):
    """Test updating an existing equation in the ModelEquation component."""
    # Initialize with ARMA equation
    model_equation.set_arma_equation(1, 1, True)
    
    # Update parameters
    parameters = {"phi_1": 0.7, "theta_1": -0.3, "constant": 0.05}
    model_equation.update_parameters(parameters)
    
    # Verify the updated equation contains the parameter values
    updated_eq = model_equation.get_latex_text()
    assert "0.700" in updated_eq
    assert "-0.300" in updated_eq
    assert "0.0500" in updated_eq

@pytest.mark.parametrize('ar_order, ma_order, constant, expected_latex', [
    (1, 0, True, "y_t = \\mu + \\phi_{1} y_{t-1} + \\varepsilon_t"),
    (0, 1, True, "y_t = \\mu + \\theta_{1} \\varepsilon_{t-1} + \\varepsilon_t"),
    (1, 1, False, "y_t = \\phi_{1} y_{t-1} + \\theta_{1} \\varepsilon_{t-1} + \\varepsilon_t"),
    (2, 2, True, "y_t = \\mu + \\phi_{1} y_{t-1} - \\phi_{2} y_{t-2} + \\theta_{1} \\varepsilon_{t-1} + \\theta_{2} \\varepsilon_{t-2} + \\varepsilon_t")
])
def test_model_equation_arma_display(model_equation, ar_order, ma_order, constant, expected_latex):
    """Test the correct display of ARMA model equations with different parameters."""
    # Set ARMA equation with given parameters
    model_equation.set_arma_equation(ar_order, ma_order, constant)
    
    # Get the rendered equation
    latex_text = model_equation.get_latex_text()
    
    # Check that key components of the expected LaTeX are present
    # We don't check the entire string as formatting might differ slightly
    for component in expected_latex.split("+"):
        component = component.strip()
        if component:
            assert component in latex_text, f"Expected '{component}' in '{latex_text}'"

@pytest.mark.parametrize('p, q, expected_latex', [
    (1, 0, "\\sigma^2_t = \\omega + \\beta_{1} \\sigma^2_{t-1}"),
    (0, 1, "\\sigma^2_t = \\omega + \\alpha_{1} \\varepsilon^2_{t-1}"),
    (1, 1, "\\sigma^2_t = \\omega + \\alpha_{1} \\varepsilon^2_{t-1} + \\beta_{1} \\sigma^2_{t-1}"),
    (2, 2, "\\sigma^2_t = \\omega + \\alpha_{1} \\varepsilon^2_{t-1} + \\alpha_{2} \\varepsilon^2_{t-2} + \\beta_{1} \\sigma^2_{t-1} + \\beta_{2} \\sigma^2_{t-2}")
])
def test_model_equation_garch_display(model_equation, p, q, expected_latex):
    """Test the correct display of GARCH model equations with different parameters."""
    # Set GARCH equation with given parameters
    model_equation.set_garch_equation(p, q)
    
    # Get the rendered equation
    latex_text = model_equation.get_latex_text()
    
    # Check that key components of the expected LaTeX are present
    # We don't check the entire string as formatting might differ slightly
    for component in expected_latex.split("+"):
        component = component.strip()
        if component:
            assert component in latex_text, f"Expected '{component}' in '{latex_text}'"

def test_model_equation_multivariate_display(model_equation):
    """Test the correct display of multivariate model equations such as BEKK, CCC, and DCC."""
    # Test BEKK equation
    bekk_eq = "H_t = C'C + A'\\varepsilon_{t-1}\\varepsilon'_{t-1}A + B'H_{t-1}B"
    model_equation.set_custom_equation(bekk_eq)
    latex_text = model_equation.get_latex_text()
    assert "H_t" in latex_text
    assert "C" in latex_text
    assert "A" in latex_text
    assert "B" in latex_text
    
    # Test CCC equation
    ccc_eq = "H_t = D_t R D_t"
    model_equation.set_custom_equation(ccc_eq)
    latex_text = model_equation.get_latex_text()
    assert "H_t" in latex_text
    assert "D_t" in latex_text
    assert "R" in latex_text
    
    # Test DCC equation
    dcc_eq = "R_t = Q_t^{*-1} Q_t Q_t^{*-1}"
    model_equation.set_custom_equation(dcc_eq)
    latex_text = model_equation.get_latex_text()
    assert "R_t" in latex_text
    assert "Q_t" in latex_text
    assert "*-1" in latex_text

def test_model_equation_resize(model_equation, app):
    """Test that the equation display adapts correctly when the component is resized."""
    # Set initial equation
    model_equation.set_arma_equation(1, 1, True)
    
    # Process events to ensure initial rendering is complete
    QTest.qWait(100)
    
    # Store initial size
    initial_size = model_equation.size()
    
    # Resize the widget
    new_width = initial_size.width() + 100
    new_height = initial_size.height() + 50
    model_equation.resize(new_width, new_height)
    
    # Process pending events to ensure resize is applied
    QTest.qWait(100)
    
    # Verify widget resized
    assert model_equation.width() == new_width
    assert model_equation.height() == new_height
    
    # Verify equation is still displayed (widget should not be empty)
    assert model_equation.latex_text is not None
    assert model_equation.latex_text != ""

def test_model_equation_error_handling(model_equation):
    """Test that the component properly handles invalid LaTeX or equation inputs."""
    # Start with a clean state
    model_equation.clear()
    assert model_equation.latex_text == ""
    
    # Test with malformed LaTeX that would cause rendering issues
    # Due to the defensive implementation, this might not crash but should handle the error
    malformed_latex = "\\frac{1}{0} + \\undefinedcommand"
    model_equation.set_custom_equation(malformed_latex)
    
    # The text should be stored even if rendering fails
    assert model_equation.latex_text != ""
    
    # Try to render an empty string
    model_equation.clear()
    model_equation.set_custom_equation("")
    assert model_equation.latex_text == "$$$"  # Empty equation gets wrapped in delimiters
    
    # Set valid equation after error to confirm recovery
    model_equation.set_arma_equation(1, 0, False)
    assert "y_t" in model_equation.latex_text
    assert "\\phi" in model_equation.latex_text