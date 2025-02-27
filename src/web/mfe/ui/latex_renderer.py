#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaTeX rendering module for MFE Toolbox UI.

This module provides functionality to render LaTeX equations for display
in the MFE Toolbox PyQt6 interface. It handles conversion of econometric
model equations to LaTeX format and renders them as images.

The module supports both synchronous and asynchronous rendering to enable
responsive UI operations during processing of complex equations.
"""

import asyncio
import logging
import io
from typing import Dict, Optional, Union, Any, Tuple

import numpy as np
from PyQt6.QtCore import QSize, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage
import matplotlib.figure as mfigure
import matplotlib.backends.backend_agg as mbackend
import matplotlib.mathtext as mathtext

# Configure logger for this module
logger = logging.getLogger(__name__)

# Default rendering parameters
DEFAULT_DPI = 100
DEFAULT_FIGSIZE = (6, 1)
DEFAULT_FONTSIZE = 12
DEFAULT_FACECOLOR = 'white'
DEFAULT_FORMAT = 'png'

def equation_to_latex(equation: str, add_delimiters: bool = True) -> str:
    """
    Convert an econometric equation to proper LaTeX format.
    
    This function takes an econometric equation and converts it to proper
    LaTeX format, handling common notation patterns and adding math delimiters
    if requested.
    
    Args:
        equation (str): The equation to convert to LaTeX
        add_delimiters (bool, optional): Whether to add LaTeX math delimiters. 
                                         Defaults to True.
    
    Returns:
        str: LaTeX formatted equation
    """
    # Check if the equation already has LaTeX delimiters
    if equation.startswith('$$') and equation.endswith('$$'):
        return equation
    if equation.startswith('\\begin{equation}') and equation.endswith('\\end{equation}'):
        return equation
    if equation.startswith('$') and equation.endswith('$'):
        return equation
    
    # Replace common econometric notation patterns
    # Replace Greek letter names with LaTeX commands
    greek_letters = {
        'alpha': '\\alpha', 'beta': '\\beta', 'gamma': '\\gamma', 'delta': '\\delta',
        'epsilon': '\\epsilon', 'varepsilon': '\\varepsilon', 'zeta': '\\zeta',
        'eta': '\\eta', 'theta': '\\theta', 'vartheta': '\\vartheta', 'iota': '\\iota',
        'kappa': '\\kappa', 'lambda': '\\lambda', 'mu': '\\mu', 'nu': '\\nu',
        'xi': '\\xi', 'pi': '\\pi', 'varpi': '\\varpi', 'rho': '\\rho',
        'varrho': '\\varrho', 'sigma': '\\sigma', 'varsigma': '\\varsigma',
        'tau': '\\tau', 'upsilon': '\\upsilon', 'phi': '\\phi', 'varphi': '\\varphi',
        'chi': '\\chi', 'psi': '\\psi', 'omega': '\\omega',
        'Alpha': '\\Alpha', 'Beta': '\\Beta', 'Gamma': '\\Gamma', 'Delta': '\\Delta',
        'Epsilon': '\\Epsilon', 'Zeta': '\\Zeta', 'Eta': '\\Eta', 'Theta': '\\Theta',
        'Iota': '\\Iota', 'Kappa': '\\Kappa', 'Lambda': '\\Lambda', 'Mu': '\\Mu',
        'Nu': '\\Nu', 'Xi': '\\Xi', 'Pi': '\\Pi', 'Rho': '\\Rho', 'Sigma': '\\Sigma',
        'Tau': '\\Tau', 'Upsilon': '\\Upsilon', 'Phi': '\\Phi', 'Chi': '\\Chi',
        'Psi': '\\Psi', 'Omega': '\\Omega'
    }
    
    # Process the equation by replacing words with Greek letters
    # Need to be careful with replacements to not replace substrings of other words
    words = equation.split()
    for i, word in enumerate(words):
        # Process pure Greek letter words
        clean_word = word.strip(',;:.()[]{}')
        if clean_word.lower() in greek_letters:
            words[i] = word.replace(clean_word, greek_letters[clean_word.lower()])
    
    equation = ' '.join(words)
    
    # Handle common econometric notation
    replacements = [
        # Operators and symbols
        ('>=', '\\geq '),
        ('<=', '\\leq '),
        ('!=', '\\neq '),
        ('->', '\\rightarrow '),
        ('<-', '\\leftarrow '),
        ('==', '='),
        
        # Statistical symbols
        ('sigma^2', '\\sigma^{2}'),
        ('var', '\\text{var}'),
        ('cov', '\\text{cov}'),
        ('corr', '\\text{corr}'),
        ('E[', '\\mathbb{E}['),
        ('Var[', '\\text{Var}['),
        ('Cov[', '\\text{Cov}['),
        ('Corr[', '\\text{Corr}['),
        
        # Econometric model notation
        ('ARMA', '\\text{ARMA}'),
        ('GARCH', '\\text{GARCH}'),
        ('ARCH', '\\text{ARCH}'),
        ('AR', '\\text{AR}'),
        ('MA', '\\text{MA}'),
        ('EGARCH', '\\text{EGARCH}'),
        ('TGARCH', '\\text{TGARCH}'),
        ('APARCH', '\\text{APARCH}'),
        
        # Matrix notation
        ('\'', '^{\\prime}'),
        ('transpose', '^{\\intercal}'),
        ('inv', '^{-1}')
    ]
    
    for old, new in replacements:
        equation = equation.replace(old, new)
    
    # Process subscripts (simple case)
    # This is a simplified approach; complex subscripts might need special handling
    import re
    # Handle subscripts: x_1 -> x_{1}, but handle x_{123} correctly
    equation = re.sub(r'([a-zA-Z])_([a-zA-Z0-9])(?!\})', r'\1_{\2}', equation)
    
    # Handle superscripts: x^2 -> x^{2}, but handle x^{123} correctly
    equation = re.sub(r'([a-zA-Z])\\?\^([a-zA-Z0-9])(?!\})', r'\1^{\2}', equation)
    
    # Add math delimiters if requested
    if add_delimiters:
        equation = f"${equation}$"
    
    return equation

def render_latex(equation: str, options: Optional[Dict[str, Any]] = None) -> io.BytesIO:
    """
    Render a LaTeX equation to an image using Matplotlib.
    
    Args:
        equation (str): The equation to render
        options (Optional[Dict[str, Any]], optional): Rendering options. Defaults to None.
    
    Returns:
        io.BytesIO: BytesIO containing the rendered image data
    """
    # Set default options if none provided
    if options is None:
        options = {}
    
    dpi = options.get('dpi', DEFAULT_DPI)
    figsize = options.get('figsize', DEFAULT_FIGSIZE)
    fontsize = options.get('fontsize', DEFAULT_FONTSIZE)
    facecolor = options.get('facecolor', DEFAULT_FACECOLOR)
    fmt = options.get('format', DEFAULT_FORMAT)
    
    # Ensure equation is in LaTeX format
    latex_eq = equation_to_latex(equation)
    
    try:
        # Create figure and canvas
        fig = mfigure.Figure(figsize=figsize, dpi=dpi)
        canvas = mbackend.FigureCanvasAgg(fig)
        
        # Set figure properties
        fig.patch.set_facecolor(facecolor)
        
        # Add text with LaTeX equation
        fig.text(0.5, 0.5, latex_eq, fontsize=fontsize, 
                horizontalalignment='center', verticalalignment='center',
                usetex=False)  # Using matplotlib's mathtext, not LaTeX
        
        # Adjust layout
        fig.tight_layout()
        
        # Create BytesIO buffer for the image
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
        
        # Close the figure to free memory
        fig.clear()
        
        # Reset buffer position
        buf.seek(0)
        
        return buf
    except Exception as e:
        logger.error(f"Error rendering LaTeX equation: {str(e)}")
        raise RuntimeError(f"Failed to render LaTeX equation: {str(e)}") from e

async def render_latex_async(equation: str, options: Optional[Dict[str, Any]] = None) -> io.BytesIO:
    """
    Asynchronously render a LaTeX equation to an image.
    
    Args:
        equation (str): The equation to render
        options (Optional[Dict[str, Any]], optional): Rendering options. Defaults to None.
    
    Returns:
        io.BytesIO: BytesIO containing the rendered image data
    """
    try:
        # Create loop for running in executor
        loop = asyncio.get_event_loop()
        
        # Run render_latex in thread pool
        result = await loop.run_in_executor(
            None, 
            lambda: render_latex(equation, options)
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in asynchronous LaTeX rendering: {str(e)}")
        raise RuntimeError(f"Failed to render LaTeX equation asynchronously: {str(e)}") from e


class LatexRenderer(QObject):
    """
    Class for rendering LaTeX equations in PyQt6 applications.
    
    This class provides methods to render LaTeX equations as QPixmap or QImage
    objects for display in PyQt6 widgets. It supports both synchronous and
    asynchronous rendering operations.
    
    Attributes:
        dpi (int): Dots per inch for rendered images
        figsize (Tuple[float, float]): Figure size in inches (width, height)
        fontsize (float): Font size for rendered equations
        facecolor (str): Background color for rendered images
        format (str): Image format for rendering ('png', 'jpg', etc.)
    """
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the LaTeX renderer with rendering options.
        
        Args:
            options (Optional[Dict[str, Any]], optional): Rendering options. Defaults to None.
        """
        super().__init__()
        
        # Set default options
        self.dpi = DEFAULT_DPI
        self.figsize = DEFAULT_FIGSIZE
        self.fontsize = DEFAULT_FONTSIZE
        self.facecolor = DEFAULT_FACECOLOR
        self.format = DEFAULT_FORMAT
        
        # Override defaults with provided options
        if options:
            self.configure(options)
        
        logger.info(f"LatexRenderer initialized with dpi={self.dpi}, "
                   f"figsize={self.figsize}, fontsize={self.fontsize}")
    
    def configure(self, options: Dict[str, Any]) -> None:
        """
        Update renderer configuration with new options.
        
        Args:
            options (Dict[str, Any]): New rendering options
        """
        if 'dpi' in options:
            self.dpi = options['dpi']
        
        if 'figsize' in options:
            self.figsize = options['figsize']
        
        if 'fontsize' in options:
            self.fontsize = options['fontsize']
        
        if 'facecolor' in options:
            self.facecolor = options['facecolor']
        
        if 'format' in options:
            self.format = options['format']
        
        logger.debug(f"LatexRenderer configuration updated: dpi={self.dpi}, "
                    f"figsize={self.figsize}, fontsize={self.fontsize}")
    
    def render_to_pixmap(self, equation: str, options: Optional[Dict[str, Any]] = None) -> QPixmap:
        """
        Render LaTeX equation to a QPixmap for display in Qt widgets.
        
        Args:
            equation (str): The equation to render
            options (Optional[Dict[str, Any]], optional): Override rendering options. Defaults to None.
        
        Returns:
            QPixmap: QPixmap containing the rendered equation
        """
        try:
            # Get rendering options, merging defaults with provided options
            render_options = self._get_render_options(options)
            
            # Render LaTeX to image data
            buffer = render_latex(equation, render_options)
            
            # Create QPixmap from image data
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            
            logger.debug(f"Rendered equation to QPixmap: {equation[:30]}{'...' if len(equation) > 30 else ''}")
            
            return pixmap
        except Exception as e:
            logger.error(f"Error rendering equation to QPixmap: {str(e)}")
            # Return an empty pixmap in case of error
            return QPixmap()
    
    def render_to_image(self, equation: str, options: Optional[Dict[str, Any]] = None) -> QImage:
        """
        Render LaTeX equation to a QImage.
        
        Args:
            equation (str): The equation to render
            options (Optional[Dict[str, Any]], optional): Override rendering options. Defaults to None.
        
        Returns:
            QImage: QImage containing the rendered equation
        """
        try:
            # Get rendering options, merging defaults with provided options
            render_options = self._get_render_options(options)
            
            # Render LaTeX to image data
            buffer = render_latex(equation, render_options)
            
            # Create QImage from image data
            image = QImage()
            image.loadFromData(buffer.getvalue())
            
            logger.debug(f"Rendered equation to QImage: {equation[:30]}{'...' if len(equation) > 30 else ''}")
            
            return image
        except Exception as e:
            logger.error(f"Error rendering equation to QImage: {str(e)}")
            # Return an empty image in case of error
            return QImage()
    
    async def render_async(self, equation: str, options: Optional[Dict[str, Any]] = None) -> QPixmap:
        """
        Asynchronously render LaTeX equation to a QPixmap.
        
        Args:
            equation (str): The equation to render
            options (Optional[Dict[str, Any]], optional): Override rendering options. Defaults to None.
        
        Returns:
            QPixmap: QPixmap containing the rendered equation
        """
        try:
            # Get rendering options, merging defaults with provided options
            render_options = self._get_render_options(options)
            
            # Asynchronously render LaTeX to image data
            buffer = await render_latex_async(equation, render_options)
            
            # Create QPixmap from image data
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            
            logger.debug(f"Asynchronously rendered equation to QPixmap: "
                        f"{equation[:30]}{'...' if len(equation) > 30 else ''}")
            
            return pixmap
        except Exception as e:
            logger.error(f"Error in asynchronous rendering: {str(e)}")
            # Return an empty pixmap in case of error
            return QPixmap()
    
    def _get_render_options(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get rendering options, merging defaults with provided options.
        
        Args:
            options (Optional[Dict[str, Any]], optional): Override options. Defaults to None.
        
        Returns:
            Dict[str, Any]: Merged options dictionary
        """
        # Start with current configuration
        render_options = {
            'dpi': self.dpi,
            'figsize': self.figsize,
            'fontsize': self.fontsize,
            'facecolor': self.facecolor,
            'format': self.format
        }
        
        # Update with provided options
        if options:
            render_options.update(options)
        
        return render_options