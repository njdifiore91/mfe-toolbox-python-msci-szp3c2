#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the LaTeX renderer component.

This module contains tests for the LaTeX renderer which provides
equation rendering capabilities for the MFE Toolbox UI.
"""

import pytest
import io
import asyncio
from unittest.mock import patch

from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QSize

from mfe.ui.latex_renderer import (
    equation_to_latex,
    render_latex,
    render_latex_async,
    LatexRenderer
)

# Sample equations for testing
SAMPLE_EQUATIONS = [
    "y_t = \\phi_1 y_{t-1} + \\varepsilon_t",
    "\\sigma_t^2 = \\omega + \\alpha_1 \\varepsilon_{t-1}^2 + \\beta_1 \\sigma_{t-1}^2",
    "E(X) = \\mu"
]

# Test options for rendering
TEST_OPTIONS = {
    "fontsize": 14,
    "dpi": 120,
    "figsize": (8, 2),
    "facecolor": "white"
}


class TestLatexRenderer:
    """Test class containing unit tests for the LaTeX renderer functionality"""
    
    def setup_method(self, method):
        """Set up test environment before each test method"""
        # Initialize test data if needed
        pass
        
    def teardown_method(self, method):
        """Clean up after each test method"""
        # Clean up resources if needed
        pass


def test_equation_to_latex():
    """Tests that equation_to_latex correctly converts equations to LaTeX format"""
    # Test basic equation without delimiters
    equation = "y = alpha + beta*x + epsilon"
    result = equation_to_latex(equation)
    assert result.startswith("$")
    assert result.endswith("$")
    assert "\\alpha" in result
    assert "\\beta" in result
    assert "\\epsilon" in result
    
    # Test equation with existing delimiters
    equation = "$y = \\alpha + \\beta x + \\epsilon$"
    result = equation_to_latex(equation)
    assert result == equation  # Should not change if already has delimiters
    
    # Test with add_delimiters=False
    equation = "y = alpha + beta*x + epsilon"
    result = equation_to_latex(equation, add_delimiters=False)
    assert not result.startswith("$")
    assert not result.endswith("$")
    assert "\\alpha" in result
    assert "\\beta" in result
    assert "\\epsilon" in result
    
    # Test Greek letter conversion
    equation = "alpha beta gamma delta epsilon"
    result = equation_to_latex(equation)
    assert "\\alpha" in result
    assert "\\beta" in result
    assert "\\gamma" in result
    assert "\\delta" in result
    assert "\\epsilon" in result
    
    # Test subscripts and superscripts
    equation = "x_1 y^2 z_i^j"
    result = equation_to_latex(equation, add_delimiters=False)
    assert "x_{1}" in result
    assert "y^{2}" in result


def test_render_latex():
    """Tests that render_latex correctly produces image data from LaTeX equations"""
    # Test basic rendering
    equation = SAMPLE_EQUATIONS[0]
    result = render_latex(equation)
    
    # Result should be a BytesIO object
    assert isinstance(result, io.BytesIO)
    
    # The BytesIO should contain image data (not empty)
    assert len(result.getvalue()) > 0
    
    # Test with custom options
    result_with_options = render_latex(equation, TEST_OPTIONS)
    assert isinstance(result_with_options, io.BytesIO)
    assert len(result_with_options.getvalue()) > 0
    
    # The image with custom options should be different from the default
    assert result.getvalue() != result_with_options.getvalue()


def test_render_latex_invalid_equation():
    """Tests error handling when rendering an invalid LaTeX equation"""
    # Test with invalid LaTeX syntax
    invalid_equation = "\\invalid{syntax"
    
    with pytest.raises(RuntimeError) as excinfo:
        render_latex(invalid_equation)
    
    # Verify exception message contains error information
    assert "Failed to render LaTeX equation" in str(excinfo.value)


def test_renderer_init():
    """Tests the initialization of the LatexRenderer class"""
    # Test with default options
    renderer = LatexRenderer()
    assert renderer.dpi == 100  # DEFAULT_DPI
    assert renderer.figsize == (6, 1)  # DEFAULT_FIGSIZE
    assert renderer.fontsize == 12  # DEFAULT_FONTSIZE
    assert renderer.facecolor == 'white'  # DEFAULT_FACECOLOR
    assert renderer.format == 'png'  # DEFAULT_FORMAT
    
    # Test with custom options
    custom_renderer = LatexRenderer(TEST_OPTIONS)
    assert custom_renderer.dpi == TEST_OPTIONS['dpi']
    assert custom_renderer.figsize == TEST_OPTIONS['figsize']
    assert custom_renderer.fontsize == TEST_OPTIONS['fontsize']
    assert custom_renderer.facecolor == TEST_OPTIONS['facecolor']


def test_renderer_configure():
    """Tests the configure method of the LatexRenderer class"""
    renderer = LatexRenderer()
    
    # Original values
    original_dpi = renderer.dpi
    original_fontsize = renderer.fontsize
    
    # New options
    new_options = {
        'dpi': original_dpi * 2,
        'fontsize': original_fontsize + 5
    }
    
    # Configure with new options
    renderer.configure(new_options)
    
    # Verify values were updated
    assert renderer.dpi == new_options['dpi']
    assert renderer.fontsize == new_options['fontsize']
    # Other values should remain unchanged
    assert renderer.format == 'png'


def test_render_to_pixmap():
    """Tests the render_to_pixmap method of the LatexRenderer class"""
    renderer = LatexRenderer()
    equation = SAMPLE_EQUATIONS[0]
    
    # Test rendering to pixmap
    pixmap = renderer.render_to_pixmap(equation)
    
    # Result should be a valid QPixmap
    assert isinstance(pixmap, QPixmap)
    assert not pixmap.isNull()
    assert pixmap.width() > 0
    assert pixmap.height() > 0
    
    # Test with custom options
    custom_pixmap = renderer.render_to_pixmap(equation, TEST_OPTIONS)
    assert isinstance(custom_pixmap, QPixmap)
    assert not custom_pixmap.isNull()


def test_render_to_image():
    """Tests the render_to_image method of the LatexRenderer class"""
    renderer = LatexRenderer()
    equation = SAMPLE_EQUATIONS[0]
    
    # Test rendering to image
    image = renderer.render_to_image(equation)
    
    # Result should be a valid QImage
    assert isinstance(image, QImage)
    assert not image.isNull()
    assert image.width() > 0
    assert image.height() > 0
    
    # Test with custom options
    custom_image = renderer.render_to_image(equation, TEST_OPTIONS)
    assert isinstance(custom_image, QImage)
    assert not custom_image.isNull()


@pytest.mark.asyncio
async def test_async_render_latex():
    """Tests asynchronous rendering of LaTeX equations"""
    equation = SAMPLE_EQUATIONS[0]
    
    # Test basic async rendering
    result = await render_latex_async(equation)
    
    # Result should be a BytesIO object
    assert isinstance(result, io.BytesIO)
    
    # The BytesIO should contain image data (not empty)
    assert len(result.getvalue()) > 0
    
    # Test with custom options
    result_with_options = await render_latex_async(equation, TEST_OPTIONS)
    assert isinstance(result_with_options, io.BytesIO)
    assert len(result_with_options.getvalue()) > 0


@pytest.mark.asyncio
async def test_async_renderer():
    """Tests the asynchronous rendering methods of the LatexRenderer class"""
    renderer = LatexRenderer()
    equation = SAMPLE_EQUATIONS[0]
    
    # Test async rendering to pixmap
    pixmap = await renderer.render_async(equation)
    
    # Result should be a valid QPixmap
    assert isinstance(pixmap, QPixmap)
    assert not pixmap.isNull()
    assert pixmap.width() > 0
    assert pixmap.height() > 0
    
    # Test with custom options
    custom_pixmap = await renderer.render_async(equation, TEST_OPTIONS)
    assert isinstance(custom_pixmap, QPixmap)
    assert not custom_pixmap.isNull()


@pytest.mark.slow
def test_renderer_performance():
    """Tests the performance of rendering operations"""
    import time
    
    renderer = LatexRenderer()
    equation = SAMPLE_EQUATIONS[1]  # Using a more complex equation
    
    # Set up timing mechanism
    start_time = time.time()
    
    # Render a complex equation multiple times
    for _ in range(5):  # Render 5 times
        pixmap = renderer.render_to_pixmap(equation)
    
    sync_time = time.time() - start_time
    
    # Verify that rendering completes within expected time limits
    assert sync_time < 10.0  # Should complete within 10 seconds
    
    # Verify pixmap is valid
    assert not pixmap.isNull()
    assert pixmap.width() > 0
    assert pixmap.height() > 0