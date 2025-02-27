#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaTeX Equation Example for MFE Toolbox.

This example demonstrates how to render and display LaTeX equations in PyQt6
applications using the MFE Toolbox's LaTeX rendering capabilities. It shows
different approaches for displaying econometric model equations, including
synchronous and asynchronous rendering.
"""

import asyncio
import logging
import random
import sys
from typing import Dict, Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QSlider, QComboBox, QTabWidget, 
    QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit
)

# Import MFE Toolbox components
from ..mfe.ui.latex_renderer import LatexRenderer
from ..mfe.ui.components.model_equation import ModelEquation

# Configure logger
logger = logging.getLogger(__name__)

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
DEFAULT_FONTSIZE = 12

# Sample equations
EXAMPLE_EQUATIONS = [
    "y_t = \\alpha + \\beta x_t + \\varepsilon_t",
    "\\sigma^2_t = \\omega + \\alpha \\varepsilon^2_{t-1} + \\beta\\sigma^2_{t-1}",
    "\\frac{1}{n}\\sum_{i=1}^{n} (x_i - \\bar{x})^2"
]

def center_window(window: QWidget) -> None:
    """
    Centers a window on the screen.
    
    Args:
        window: The window to center
    """
    # Get the screen geometry
    screen_geometry = QApplication.primaryScreen().geometry()
    
    # Calculate center point
    center_point = screen_geometry.center()
    
    # Get window geometry
    window_geometry = window.frameGeometry()
    
    # Move window to center
    window_geometry.moveCenter(center_point)
    window.move(window_geometry.topLeft())
    
    logger.debug("Window centered on screen")

def generate_random_parameters(model_type: str) -> Dict[str, float]:
    """
    Generates random parameters for example equations.
    
    Args:
        model_type: Type of model ('ARMA', 'GARCH', or 'Custom')
        
    Returns:
        Dictionary of parameter names and values
    """
    parameters = {}
    
    if model_type == 'ARMA':
        # Generate AR coefficients
        parameters['phi_1'] = random.uniform(-0.8, 0.8)
        parameters['phi_2'] = random.uniform(-0.4, 0.4)
        
        # Generate MA coefficients
        parameters['theta_1'] = random.uniform(-0.8, 0.8)
        
        # Generate constant
        parameters['constant'] = random.uniform(-0.1, 0.1)
        
    elif model_type == 'GARCH':
        # Generate GARCH parameters
        parameters['omega'] = random.uniform(0.01, 0.1)
        parameters['alpha_1'] = random.uniform(0.05, 0.2)
        parameters['beta_1'] = random.uniform(0.7, 0.9)
        
    else:  # Custom
        # Generate generic parameters
        parameters['alpha'] = random.uniform(0.1, 0.9)
        parameters['beta'] = random.uniform(0.1, 0.9)
    
    logger.debug(f"Generated random parameters for {model_type} model")
    return parameters

class DirectRenderingTab(QWidget):
    """
    Tab demonstrating direct LaTeX rendering using LatexRenderer.
    
    This tab shows how to directly use the LatexRenderer class to render
    equations and display them in a QLabel.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the direct rendering tab.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Set up the main layout
        self.layout = QVBoxLayout(self)
        
        # Add controls section
        controls_widget = self.setup_controls()
        self.layout.addWidget(controls_widget)
        
        # Add equation display label
        self.equation_label = QLabel()
        self.equation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.equation_label)
        
        # Initialize renderer
        self.renderer = LatexRenderer()
        
        # Set initial values
        self.current_index = 0
        self.current_fontsize = DEFAULT_FONTSIZE
        
        # Set initial equation
        self.on_equation_changed(0)
    
    def setup_controls(self) -> QWidget:
        """
        Sets up the control widgets for equation selection and font size.
        
        Returns:
            Widget containing the controls
        """
        # Create container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Create equation selection combo box
        equation_layout = QHBoxLayout()
        equation_layout.addWidget(QLabel("Equation:"))
        
        self.equation_combo = QComboBox()
        for eq in EXAMPLE_EQUATIONS:
            self.equation_combo.addItem(eq)
        equation_layout.addWidget(self.equation_combo)
        layout.addLayout(equation_layout)
        
        # Create font size slider
        fontsize_layout = QHBoxLayout()
        fontsize_layout.addWidget(QLabel("Font Size:"))
        
        self.font_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_size_slider.setMinimum(8)
        self.font_size_slider.setMaximum(24)
        self.font_size_slider.setValue(DEFAULT_FONTSIZE)
        fontsize_layout.addWidget(self.font_size_slider)
        fontsize_layout.addWidget(QLabel(f"{DEFAULT_FONTSIZE}pt"))
        layout.addLayout(fontsize_layout)
        
        # Connect signals to slots
        self.equation_combo.currentIndexChanged.connect(self.on_equation_changed)
        self.font_size_slider.valueChanged.connect(self.on_fontsize_changed)
        
        return container
    
    def on_equation_changed(self, index: int) -> None:
        """
        Handle equation selection change.
        
        Args:
            index: Index of the selected equation
        """
        self.current_index = index
        equation = self.equation_combo.itemText(index)
        self.render_equation(equation, self.current_fontsize)
    
    def on_fontsize_changed(self, value: int) -> None:
        """
        Handle font size slider change.
        
        Args:
            value: New font size value
        """
        self.current_fontsize = value
        equation = self.equation_combo.itemText(self.current_index)
        self.render_equation(equation, self.current_fontsize)
    
    def render_equation(self, equation: str, fontsize: float) -> None:
        """
        Render the equation using direct LatexRenderer access.
        
        Args:
            equation: LaTeX equation to render
            fontsize: Font size for rendering
        """
        # Create render options
        options = {
            'fontsize': fontsize,
        }
        
        try:
            # Render equation to QPixmap
            pixmap = self.renderer.render_to_pixmap(equation, options)
            
            # Display pixmap in label
            self.equation_label.setPixmap(pixmap)
            
            logger.debug(f"Rendered equation: {equation[:30]}{'...' if len(equation) > 30 else ''}")
        except Exception as e:
            logger.error(f"Error rendering equation: {str(e)}")
            self.equation_label.setText(f"Error rendering equation: {str(e)}")
    
    async def render_equation_async(self, equation: str, fontsize: float) -> None:
        """
        Asynchronously render the equation to avoid blocking UI.
        
        Args:
            equation: LaTeX equation to render
            fontsize: Font size for rendering
        """
        # Create render options
        options = {
            'fontsize': fontsize,
        }
        
        try:
            # Render equation asynchronously
            pixmap = await self.renderer.render_async(equation, options)
            
            # Display pixmap in label
            self.equation_label.setPixmap(pixmap)
            
            logger.debug(f"Asynchronously rendered equation: {equation[:30]}{'...' if len(equation) > 30 else ''}")
        except Exception as e:
            logger.error(f"Error in asynchronous rendering: {str(e)}")
            self.equation_label.setText(f"Error rendering equation: {str(e)}")

class ModelEquationTab(QWidget):
    """
    Tab demonstrating ModelEquation widget for econometric model equations.
    
    This tab shows how to use the ModelEquation widget to display different
    types of econometric models with proper formatting and parameter updates.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the ModelEquation tab.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Set up the main layout
        self.layout = QVBoxLayout(self)
        
        # Add controls section
        controls_widget = self.setup_controls()
        self.layout.addWidget(controls_widget)
        
        # Add equation widget
        self.equation_widget = ModelEquation()
        self.layout.addWidget(self.equation_widget)
        
        # Set initial values
        self.current_model_type = 'ARMA'
        self.current_parameters = {}
        
        # Set initial equation
        self.update_equation()
    
    def setup_controls(self) -> QWidget:
        """
        Sets up the control widgets for model configuration.
        
        Returns:
            Widget containing the controls
        """
        # Create container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Create model type selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Type:"))
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(['ARMA', 'GARCH', 'Custom'])
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Create ARMA controls
        arma_layout = QHBoxLayout()
        arma_layout.addWidget(QLabel("AR Order:"))
        self.ar_order_spin = QSpinBox()
        self.ar_order_spin.setRange(0, 5)
        self.ar_order_spin.setValue(2)
        arma_layout.addWidget(self.ar_order_spin)
        
        arma_layout.addWidget(QLabel("MA Order:"))
        self.ma_order_spin = QSpinBox()
        self.ma_order_spin.setRange(0, 5)
        self.ma_order_spin.setValue(1)
        arma_layout.addWidget(self.ma_order_spin)
        
        self.include_constant_check = QCheckBox("Include Constant")
        self.include_constant_check.setChecked(True)
        arma_layout.addWidget(self.include_constant_check)
        layout.addLayout(arma_layout)
        
        # Create GARCH controls
        garch_layout = QHBoxLayout()
        garch_layout.addWidget(QLabel("GARCH Order (p):"))
        self.garch_p_spin = QSpinBox()
        self.garch_p_spin.setRange(0, 5)
        self.garch_p_spin.setValue(1)
        garch_layout.addWidget(self.garch_p_spin)
        
        garch_layout.addWidget(QLabel("ARCH Order (q):"))
        self.garch_q_spin = QSpinBox()
        self.garch_q_spin.setRange(0, 5)
        self.garch_q_spin.setValue(1)
        garch_layout.addWidget(self.garch_q_spin)
        layout.addLayout(garch_layout)
        
        # Update parameters button
        self.update_params_button = QPushButton("Generate Random Parameters")
        layout.addWidget(self.update_params_button)
        
        # Connect signals
        self.model_combo.currentIndexChanged.connect(self.on_model_type_changed)
        self.update_params_button.clicked.connect(self.on_parameters_changed)
        self.ar_order_spin.valueChanged.connect(self.update_equation)
        self.ma_order_spin.valueChanged.connect(self.update_equation)
        self.include_constant_check.stateChanged.connect(self.update_equation)
        self.garch_p_spin.valueChanged.connect(self.update_equation)
        self.garch_q_spin.valueChanged.connect(self.update_equation)
        
        # Show/hide appropriate controls
        self.on_model_type_changed(0)
        
        return container
    
    def on_model_type_changed(self, index: int) -> None:
        """
        Handle model type selection change.
        
        Args:
            index: Index of the selected model type
        """
        # Get selected model type
        self.current_model_type = self.model_combo.itemText(index)
        
        # Show/hide appropriate controls
        if self.current_model_type == 'ARMA':
            self.ar_order_spin.setVisible(True)
            self.ma_order_spin.setVisible(True)
            self.include_constant_check.setVisible(True)
            self.garch_p_spin.setVisible(False)
            self.garch_q_spin.setVisible(False)
        elif self.current_model_type == 'GARCH':
            self.ar_order_spin.setVisible(False)
            self.ma_order_spin.setVisible(False)
            self.include_constant_check.setVisible(False)
            self.garch_p_spin.setVisible(True)
            self.garch_q_spin.setVisible(True)
        else:  # Custom
            self.ar_order_spin.setVisible(False)
            self.ma_order_spin.setVisible(False)
            self.include_constant_check.setVisible(False)
            self.garch_p_spin.setVisible(False)
            self.garch_q_spin.setVisible(False)
        
        # Update equation for the new model type
        self.update_equation()
    
    def on_parameters_changed(self) -> None:
        """
        Update the model with new parameters.
        """
        # Generate random parameters for the current model type
        self.current_parameters = generate_random_parameters(self.current_model_type)
        
        # Update the model equation with the new parameters
        if self.current_model_type == 'ARMA':
            self.equation_widget.update_parameters(self.current_parameters)
        elif self.current_model_type == 'GARCH':
            self.equation_widget.update_parameters(self.current_parameters)
        else:  # Custom
            # For custom, we need to update the whole equation
            self.update_equation()
        
        logger.debug(f"Updated model parameters: {', '.join(f'{k}={v:.3f}' for k, v in self.current_parameters.items())}")
    
    def update_equation(self) -> None:
        """
        Update the equation display based on current settings.
        """
        if self.current_model_type == 'ARMA':
            # Get ARMA parameters
            p = self.ar_order_spin.value()
            q = self.ma_order_spin.value()
            include_constant = self.include_constant_check.isChecked()
            
            # Update equation
            self.equation_widget.set_arma_equation(p, q, include_constant, self.current_parameters)
            
        elif self.current_model_type == 'GARCH':
            # Get GARCH parameters
            p = self.garch_p_spin.value()
            q = self.garch_q_spin.value()
            
            # Update equation
            self.equation_widget.set_garch_equation(p, q, self.current_parameters)
            
        else:  # Custom
            # Use the first example equation for custom
            custom_eq = EXAMPLE_EQUATIONS[0]
            
            # Apply any parameters
            if self.current_parameters:
                # Simple placeholder replacement
                for param, value in self.current_parameters.items():
                    custom_eq = custom_eq.replace(f"\\{param}", f"{value:.3f}")
            
            # Update equation
            self.equation_widget.set_custom_equation(custom_eq)

class AsyncRenderingTab(QWidget):
    """
    Tab demonstrating asynchronous LaTeX rendering to prevent UI blocking.
    
    This tab shows how to use asynchronous rendering to handle complex
    equations without blocking the UI thread.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the asynchronous rendering tab.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Set up the main layout
        self.layout = QVBoxLayout(self)
        
        # Add equation input
        self.layout.addWidget(QLabel("Custom LaTeX Equation:"))
        self.equation_edit = QTextEdit()
        self.equation_edit.setMinimumHeight(100)
        self.layout.addWidget(self.equation_edit)
        
        # Add render button
        self.render_button = QPushButton("Render Equation (Async)")
        self.layout.addWidget(self.render_button)
        
        # Add status label
        self.status_label = QLabel("Ready")
        self.layout.addWidget(self.status_label)
        
        # Add equation display label
        self.equation_label = QLabel()
        self.equation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.equation_label)
        
        # Initialize renderer
        self.renderer = LatexRenderer()
        
        # Initialize current task
        self.current_task = None
        
        # Connect signals
        self.render_button.clicked.connect(self.on_render_clicked)
        
        # Set default equation
        self.equation_edit.setText("\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}")
    
    def on_render_clicked(self) -> None:
        """
        Handle render button click.
        """
        # Get equation text
        equation = self.equation_edit.toPlainText()
        
        # Update status
        self.status_label.setText("Rendering...")
        
        # Cancel any existing task
        if self.current_task is not None and not self.current_task.done():
            self.current_task.cancel()
        
        # Disable button while rendering
        self.render_button.setEnabled(False)
        
        # Create task for async rendering
        loop = asyncio.get_event_loop()
        self.current_task = asyncio.create_task(self._render_equation_async(equation))
        self.current_task.add_done_callback(lambda _: self.on_render_complete())
    
    async def _render_equation_async(self, equation: str) -> QPixmap:
        """
        Asynchronously render the equation.
        
        Args:
            equation: LaTeX equation to render
            
        Returns:
            Rendered equation as QPixmap
        """
        # Create render options
        options = {
            'fontsize': DEFAULT_FONTSIZE,
        }
        
        # Simulate complex rendering with a delay
        await asyncio.sleep(1)
        
        # Render equation asynchronously
        pixmap = await self.renderer.render_async(equation, options)
        
        return pixmap
    
    def on_render_complete(self) -> None:
        """
        Handle completed rendering task.
        """
        try:
            # Get the result
            pixmap = self.current_task.result()
            
            # Display pixmap in label
            self.equation_label.setPixmap(pixmap)
            
            # Update status
            self.status_label.setText("Render complete")
            
        except Exception as e:
            # Handle rendering error
            logger.error(f"Error in asynchronous rendering: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.equation_label.setText(f"Error rendering equation: {str(e)}")
        
        # Re-enable render button
        self.render_button.setEnabled(True)
        
        # Clear task reference
        self.current_task = None

class LatexEquationExample(QMainWindow):
    """
    Main example window demonstrating different LaTeX equation rendering approaches.
    
    This window contains tabs for different LaTeX rendering methods:
    - Direct rendering using LatexRenderer
    - ModelEquation widget for econometric equations
    - Asynchronous rendering for non-blocking UI
    """
    
    def __init__(self):
        """
        Initialize the example window.
        """
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("MFE Toolbox - LaTeX Equation Example")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create and add tabs
        self.direct_tab = DirectRenderingTab()
        self.tab_widget.addTab(self.direct_tab, "Direct Rendering")
        
        self.model_tab = ModelEquationTab()
        self.tab_widget.addTab(self.model_tab, "ModelEquation Widget")
        
        self.async_tab = AsyncRenderingTab()
        self.tab_widget.addTab(self.async_tab, "Async Rendering")
        
        # Add tab widget to layout
        self.layout.addWidget(self.tab_widget)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Center window on screen
        center_window(self)
        
        logger.debug("LatexEquationExample window initialized")

def main() -> int:
    """
    Main function to initialize and run the example application.
    
    Returns:
        Application exit code
    """
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create main window
    window = LatexEquationExample()
    window.show()
    
    # Run application
    logger.info("Starting LaTeX equation example application")
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())