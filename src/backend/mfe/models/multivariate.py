"""
MFE Toolbox - Multivariate Volatility Models

This module provides the core implementation of multivariate volatility models
for analyzing and forecasting covariances and correlations in multivariate financial time series.
It includes base classes, enumerations, data structures, and shared functionality for
multivariate GARCH models like BEKK, CCC, and DCC.
"""

import abc  # Python 3.12
import enum  # Python 3.12
import logging  # Python 3.12
from dataclasses import dataclass  # Python 3.12
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Type, Union  # Python 3.12

import numpy as np  # numpy 1.26.3
import scipy.linalg  # scipy 1.11.4
import numba  # numba 0.59.0

# Internal imports
from ..models.volatility import MultivariateVolatilityModel as BaseMultivariateModel
from ..utils.validation import validate_array, is_positive_definite, is_symmetric
from ..utils.numba_helpers import optimized_jit
from ..core.optimization import Optimizer
from ..core.distributions import SkewedTDistribution
from ..utils.async_helpers import handle_exceptions_async, async_generator

# Set up module logger
logger = logging.getLogger(__name__)

# Global registry for multivariate models
MULTIVARIATE_MODELS: Dict[str, Type["MultivariateVolatilityModel"]] = {}


class MultivariateType(enum.Enum):
    """
    Enumeration of supported multivariate volatility model types.
    """
    BEKK = "BEKK"  # Baba-Engle-Kraft-Kroner model
    CCC = "CCC"    # Constant Conditional Correlation model
    DCC = "DCC"    # Dynamic Conditional Correlation model
    RARCH = "RARCH"  # Rotated ARCH model
    RCC = "RCC"    # Rotated Conditional Correlation model


class DistributionType(enum.Enum):
    """
    Enumeration of supported distributions for multivariate innovations.
    """
    NORMAL = "NORMAL"  # Multivariate normal distribution
    STUDENT = "STUDENT"  # Multivariate Student's t-distribution
    SKEWT = "SKEWT"  # Multivariate skewed t-distribution


@dataclass
class MultivariateVolatilityParameters:
    """
    Data class for parameters of multivariate volatility models.
    """
    model_type: str
    model_params: Dict[str, Any]
    initial_covariance: np.ndarray
    dist_type: DistributionType = DistributionType.NORMAL
    dist_params: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize parameter container with validation."""
        # Validate model_type is a supported type
        if not any(self.model_type == model.value for model in MultivariateType):
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Validate initial_covariance is symmetric and positive definite
        if not is_symmetric(self.initial_covariance, raise_error=False):
            raise ValueError("Initial covariance matrix must be symmetric")
        
        if not is_positive_definite(self.initial_covariance, raise_error=False):
            raise ValueError("Initial covariance matrix must be positive definite")
        
        # Initialize dist_params if None
        if self.dist_params is None:
            self.dist_params = {}
            
        # Set default distribution parameters based on distribution type
        if self.dist_type == DistributionType.NORMAL:
            # No additional parameters needed for normal distribution
            pass
        elif self.dist_type == DistributionType.STUDENT:
            # Default degrees of freedom for Student's t
            if 'df' not in self.dist_params:
                self.dist_params['df'] = 8.0
        elif self.dist_type == DistributionType.SKEWT:
            # Default parameters for skewed t
            if 'df' not in self.dist_params:
                self.dist_params['df'] = 8.0
            if 'lambda' not in self.dist_params:
                self.dist_params['lambda'] = 0.0  # Symmetric by default

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to dictionary format.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of parameters
        """
        return {
            'model_type': self.model_type,
            'model_params': self.model_params.copy(),
            'initial_covariance': self.initial_covariance.copy(),
            'dist_type': self.dist_type.value,
            'dist_params': self.dist_params.copy() if self.dist_params else {}
        }
    
    @classmethod
    def from_dict(cls, param_dict: Dict[str, Any]) -> 'MultivariateVolatilityParameters':
        """
        Create parameters object from dictionary.
        
        Parameters
        ----------
        param_dict : Dict[str, Any]
            Dictionary with parameter values
            
        Returns
        -------
        MultivariateVolatilityParameters
            Parameters object
        """
        # Convert dist_type string to enum value
        dist_type_str = param_dict.get('dist_type', 'NORMAL')
        dist_type = DistributionType[dist_type_str] if isinstance(dist_type_str, str) else dist_type_str
        
        # Create and return object
        return cls(
            model_type=param_dict['model_type'],
            model_params=param_dict['model_params'].copy(),
            initial_covariance=param_dict['initial_covariance'].copy(),
            dist_type=dist_type,
            dist_params=param_dict.get('dist_params', {}).copy()
        )
    
    def validate(self) -> bool:
        """
        Validate parameters satisfy model constraints.
        
        Returns
        -------
        bool
            True if parameters are valid
        """
        try:
            # Check model-specific constraints based on model type
            if self.model_type == MultivariateType.BEKK.value:
                # For BEKK, check required parameters
                if 'C' not in self.model_params:
                    return False
                if 'A' not in self.model_params:
                    return False
                if 'B' not in self.model_params:
                    return False
                
                # Verify dimensions are compatible
                n = self.initial_covariance.shape[0]
                C = self.model_params['C']
                A = self.model_params['A']
                B = self.model_params['B']
                
                if C.shape != (n, n) or A.shape != (n, n) or B.shape != (n, n):
                    return False
                
                # Check if C is lower triangular or symmetric
                C_is_valid = False
                # Check if all above diagonal elements are zero (lower triangular)
                if np.allclose(np.triu(C, k=1), 0):
                    C_is_valid = True
                # Or check if symmetric
                elif is_symmetric(C, raise_error=False):
                    C_is_valid = True
                
                if not C_is_valid:
                    return False
                
            elif self.model_type == MultivariateType.CCC.value:
                # For CCC, check required parameters
                if 'R' not in self.model_params:
                    return False
                if 'garch_params' not in self.model_params:
                    return False
                
                # Verify R is a valid correlation matrix
                R = self.model_params['R']
                n = self.initial_covariance.shape[0]
                
                if R.shape != (n, n):
                    return False
                
                if not is_positive_definite(R, raise_error=False):
                    return False
                
                # Check if diagonal elements of R are 1
                if not np.allclose(np.diag(R), 1.0):
                    return False
                
            elif self.model_type == MultivariateType.DCC.value:
                # For DCC, check required parameters
                if 'a' not in self.model_params:
                    return False
                if 'b' not in self.model_params:
                    return False
                if 'R_bar' not in self.model_params:
                    return False
                if 'garch_params' not in self.model_params:
                    return False
                
                # Verify R_bar is a valid correlation matrix
                R_bar = self.model_params['R_bar']
                n = self.initial_covariance.shape[0]
                
                if R_bar.shape != (n, n):
                    return False
                
                if not is_positive_definite(R_bar, raise_error=False):
                    return False
                
                # Check if diagonal elements of R_bar are 1
                if not np.allclose(np.diag(R_bar), 1.0):
                    return False
                
                # Check a and b satisfy stationarity constraint
                a = self.model_params['a']
                b = self.model_params['b']
                
                if a < 0 or b < 0 or a + b >= 1:
                    return False
            
            # Validate distribution parameters
            if self.dist_type == DistributionType.STUDENT:
                if 'df' not in self.dist_params:
                    return False
                df = self.dist_params['df']
                if df <= 2:  # df must be > 2 for finite variance
                    return False
                
            elif self.dist_type == DistributionType.SKEWT:
                if 'df' not in self.dist_params:
                    return False
                if 'lambda' not in self.dist_params:
                    return False
                
                df = self.dist_params['df']
                lambda_ = self.dist_params['lambda']
                
                if df <= 2:  # df must be > 2 for finite variance
                    return False
                
                if abs(lambda_) >= 1:  # lambda must be in (-1,1)
                    return False
            
            # All validations passed
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return False


@dataclass
class MultivariateVolatilityResult:
    """
    Container for estimation results of multivariate volatility models.
    """
    model_type: str
    parameters: MultivariateVolatilityParameters
    covariances: np.ndarray
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    log_likelihood: float
    information_criteria: Dict[str, float]
    diagnostics: Dict[str, Any]
    optimization_result: Dict[str, Any]
    
    def __post_init__(self):
        """
        Initialize result container with validation.
        """
        # Validate data dimensions are consistent
        n_obs, n_assets = self.residuals.shape
        
        # Covariances should be a 3D array with shape (T, N, N)
        if self.covariances.ndim != 3:
            raise ValueError(f"Covariances must be a 3D array, got {self.covariances.ndim}D")
        
        if self.covariances.shape[0] != n_obs:
            raise ValueError(f"Covariances must have the same number of observations as residuals")
        
        if self.covariances.shape[1] != n_assets or self.covariances.shape[2] != n_assets:
            raise ValueError(f"Covariances must have dimensions compatible with residuals")
        
        # Standardized residuals should match residuals shape
        if self.standardized_residuals.shape != self.residuals.shape:
            raise ValueError(f"Standardized residuals must have the same shape as residuals")
    
    def summary(self) -> str:
        """
        Generate a summary of estimation results.
        
        Returns
        -------
        str
            Formatted summary string
        """
        # Create summary header
        summary_lines = [
            f"Multivariate Volatility Model: {self.model_type}",
            "=" * 60,
            "",
            f"Number of assets: {self.residuals.shape[1]}",
            f"Number of observations: {self.residuals.shape[0]}",
            f"Log-likelihood: {self.log_likelihood:.4f}",
            f"AIC: {self.information_criteria.get('aic', float('nan')):.4f}",
            f"BIC: {self.information_criteria.get('bic', float('nan')):.4f}",
            "",
            "Parameter Estimates:",
            "-" * 30
        ]
        
        # Add parameters based on model type
        if self.model_type == MultivariateType.BEKK.value:
            # Extract BEKK parameters
            C = self.parameters.model_params['C']
            A = self.parameters.model_params['A']
            B = self.parameters.model_params['B']
            
            summary_lines.append("\nC (Constant term):")
            for i in range(C.shape[0]):
                summary_lines.append("  " + " ".join(f"{val:.4f}" for val in C[i, :]))
            
            summary_lines.append("\nA (ARCH effects):")
            for i in range(A.shape[0]):
                summary_lines.append("  " + " ".join(f"{val:.4f}" for val in A[i, :]))
            
            summary_lines.append("\nB (GARCH effects):")
            for i in range(B.shape[0]):
                summary_lines.append("  " + " ".join(f"{val:.4f}" for val in B[i, :]))
                
        elif self.model_type == MultivariateType.CCC.value:
            # Extract CCC parameters
            R = self.parameters.model_params['R']
            garch_params = self.parameters.model_params['garch_params']
            
            summary_lines.append("\nConstant Correlation Matrix (R):")
            for i in range(R.shape[0]):
                summary_lines.append("  " + " ".join(f"{val:.4f}" for val in R[i, :]))
            
            summary_lines.append("\nUniviariate GARCH Parameters:")
            for i, params in enumerate(garch_params):
                summary_lines.append(f"  Asset {i+1}: {params}")
                
        elif self.model_type == MultivariateType.DCC.value:
            # Extract DCC parameters
            a = self.parameters.model_params['a']
            b = self.parameters.model_params['b']
            R_bar = self.parameters.model_params['R_bar']
            garch_params = self.parameters.model_params['garch_params']
            
            summary_lines.append(f"\nDCC Parameters:")
            summary_lines.append(f"  a: {a:.4f}")
            summary_lines.append(f"  b: {b:.4f}")
            summary_lines.append(f"  Persistence (a+b): {a+b:.4f}")
            
            summary_lines.append("\nUnconditional Correlation Matrix (R_bar):")
            for i in range(R_bar.shape[0]):
                summary_lines.append("  " + " ".join(f"{val:.4f}" for val in R_bar[i, :]))
            
            summary_lines.append("\nUniviariate GARCH Parameters:")
            for i, params in enumerate(garch_params):
                summary_lines.append(f"  Asset {i+1}: {params}")
        
        # Add diagnostics
        summary_lines.extend([
            "",
            "Diagnostics:",
            "-" * 30
        ])
        
        for key, value in self.diagnostics.items():
            if isinstance(value, dict):
                summary_lines.append(f"\n{key}:")
                for sub_key, sub_value in value.items():
                    summary_lines.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, (float, int)):
                summary_lines.append(f"{key}: {value:.4f}")
            else:
                summary_lines.append(f"{key}: {value}")
        
        return "\n".join(summary_lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of results
        """
        return {
            'model_type': self.model_type,
            'parameters': self.parameters.to_dict(),
            'covariances': self.covariances.copy(),
            'residuals': self.residuals.copy(),
            'standardized_residuals': self.standardized_residuals.copy(),
            'log_likelihood': self.log_likelihood,
            'information_criteria': self.information_criteria.copy(),
            'diagnostics': self.diagnostics.copy(),
            'optimization_result': self.optimization_result.copy()
        }


@dataclass
class MultivariateVolatilityForecast:
    """
    Container for forecast results of multivariate volatility models.
    """
    model_type: str
    forecast_covariances: np.ndarray
    forecast_lower: Optional[np.ndarray] = None
    forecast_upper: Optional[np.ndarray] = None
    alpha_level: Optional[float] = None
    horizon: int = 1
    forecast_info: Dict[str, Any] = None
    
    def __post_init__(self):
        """
        Initialize forecast container with validation.
        """
        # Validate forecast_covariances is a 3D array with shape (h, N, N)
        if self.forecast_covariances.ndim != 3:
            raise ValueError(f"Forecast covariances must be a 3D array, got {self.forecast_covariances.ndim}D")
        
        h, n, m = self.forecast_covariances.shape
        
        # Check for square matrices in the last two dimensions
        if n != m:
            raise ValueError(f"Forecast covariance matrices must be square, got shape {n}x{m}")
        
        # Check for compatible forecast bounds if provided
        if self.forecast_lower is not None and self.forecast_upper is not None:
            if self.forecast_lower.shape != self.forecast_covariances.shape:
                raise ValueError(f"Lower bounds shape {self.forecast_lower.shape} doesn't match forecast shape {self.forecast_covariances.shape}")
            
            if self.forecast_upper.shape != self.forecast_covariances.shape:
                raise ValueError(f"Upper bounds shape {self.forecast_upper.shape} doesn't match forecast shape {self.forecast_covariances.shape}")
        
        # Initialize forecast_info if None
        if self.forecast_info is None:
            self.forecast_info = {}
    
    def get_forecast(self, h: int) -> Dict[str, np.ndarray]:
        """
        Get forecast values for a specific horizon.
        
        Parameters
        ----------
        h : int
            Forecast horizon (1-based indexing, h=1 is one-step ahead)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing forecast components for the specified horizon
        """
        # Validate h is within forecast horizon
        if h < 1 or h > self.forecast_covariances.shape[0]:
            raise ValueError(f"Horizon h must be between 1 and {self.forecast_covariances.shape[0]}, got {h}")
        
        # Adjust for 0-based indexing in arrays
        h_idx = h - 1
        
        # Create result dictionary
        result = {
            'covariance': self.forecast_covariances[h_idx].copy()
        }
        
        # Add interval forecasts if available
        if self.forecast_lower is not None:
            result['lower'] = self.forecast_lower[h_idx].copy()
        
        if self.forecast_upper is not None:
            result['upper'] = self.forecast_upper[h_idx].copy()
        
        # Add confidence level if available
        if self.alpha_level is not None:
            result['alpha'] = self.alpha_level
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert forecast to dictionary format.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of forecast
        """
        result = {
            'model_type': self.model_type,
            'forecast_covariances': self.forecast_covariances.copy(),
            'horizon': self.horizon,
            'forecast_info': self.forecast_info.copy() if self.forecast_info else {}
        }
        
        # Add optional components if available
        if self.forecast_lower is not None:
            result['forecast_lower'] = self.forecast_lower.copy()
        
        if self.forecast_upper is not None:
            result['forecast_upper'] = self.forecast_upper.copy()
        
        if self.alpha_level is not None:
            result['alpha_level'] = self.alpha_level
        
        return result


class MultivariateVolatilityModel(abc.ABC):
    """
    Abstract base class for all multivariate volatility models.
    """
    
    def __init__(
        self,
        n_assets: int,
        model_type: Optional[str] = None,
        dist_type: Optional[DistributionType] = None,
        dist_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base multivariate volatility model.
        
        Parameters
        ----------
        n_assets : int
            Number of assets (dimension of multivariate time series)
        model_type : Optional[str], default=None
            Model type identifier (should be overridden by subclasses)
        dist_type : Optional[DistributionType], default=None
            Distribution type for innovations
        dist_params : Optional[Dict[str, Any]], default=None
            Distribution parameters
        """
        # Validate inputs
        if n_assets <= 0:
            raise ValueError(f"Number of assets must be positive, got {n_assets}")
        
        self.n_assets = n_assets
        
        # Set model type (should be overridden by subclasses)
        self.model_type = model_type
        
        # Set default distribution type if not provided
        self.dist_type = dist_type if dist_type is not None else DistributionType.NORMAL
        
        # Initialize distribution parameters
        self.dist_params = dist_params if dist_params is not None else {}
        
        # Initialize parameters and results to None
        self.parameters = None
        self.result = None
        
        # Initialize optimizer
        self.optimizer = Optimizer()
    
    @abc.abstractmethod
    def fit(
        self,
        returns: np.ndarray,
        starting_params: Optional[MultivariateVolatilityParameters] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> MultivariateVolatilityResult:
        """
        Fit model to multivariate return data.
        
        Parameters
        ----------
        returns : np.ndarray
            Multivariate return series with shape (T, N)
        starting_params : Optional[MultivariateVolatilityParameters], default=None
            Initial parameter values for optimization
        options : Optional[Dict[str, Any]], default=None
            Options for optimization
            
        Returns
        -------
        MultivariateVolatilityResult
            Estimation results
        """
        pass
    
    @abc.abstractmethod
    async def fit_async(
        self,
        returns: np.ndarray,
        starting_params: Optional[MultivariateVolatilityParameters] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Coroutine[Any, Any, MultivariateVolatilityResult]:
        """
        Asynchronously fit model to multivariate return data.
        
        Parameters
        ----------
        returns : np.ndarray
            Multivariate return series with shape (T, N)
        starting_params : Optional[MultivariateVolatilityParameters], default=None
            Initial parameter values for optimization
        options : Optional[Dict[str, Any]], default=None
            Options for optimization
            
        Returns
        -------
        Coroutine[Any, Any, MultivariateVolatilityResult]
            Coroutine yielding estimation result
        """
        pass
    
    @abc.abstractmethod
    def forecast(
        self,
        horizon: int,
        returns: Optional[np.ndarray] = None,
        covariances: Optional[np.ndarray] = None,
        alpha_level: Optional[float] = None,
        n_simulations: Optional[int] = None
    ) -> MultivariateVolatilityForecast:
        """
        Forecast conditional covariance matrices.
        
        Parameters
        ----------
        horizon : int
            Forecast horizon
        returns : Optional[np.ndarray], default=None
            Historical return series. If None, uses the data from fitting.
        covariances : Optional[np.ndarray], default=None
            Historical covariance matrices. If None, recomputed from model.
        alpha_level : Optional[float], default=None
            Confidence level for forecast intervals
        n_simulations : Optional[int], default=None
            Number of simulations for Monte Carlo forecast intervals
            
        Returns
        -------
        MultivariateVolatilityForecast
            Forecast results
        """
        pass
    
    @abc.abstractmethod
    def simulate(
        self,
        n_obs: int,
        initial_values: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate multivariate returns.
        
        Parameters
        ----------
        n_obs : int
            Number of observations to simulate
        initial_values : Optional[np.ndarray], default=None
            Initial values to start the simulation
        rng : Optional[np.random.Generator], default=None
            Random number generator
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Simulated returns and covariances
        """
        pass
    
    @abc.abstractmethod
    def filter_covariances(
        self,
        returns: np.ndarray,
        params: MultivariateVolatilityParameters
    ) -> np.ndarray:
        """
        Filter historical covariance matrices.
        
        Parameters
        ----------
        returns : np.ndarray
            Multivariate return series with shape (T, N)
        params : MultivariateVolatilityParameters
            Model parameters
            
        Returns
        -------
        np.ndarray
            Conditional covariance matrices with shape (T, N, N)
        """
        pass
    
    def validate_returns(self, returns: np.ndarray) -> np.ndarray:
        """
        Validate multivariate returns data.
        
        Parameters
        ----------
        returns : np.ndarray
            Multivariate return series
            
        Returns
        -------
        np.ndarray
            Validated returns
        """
        # Validate returns is a 2D array
        returns = validate_array(returns, param_name="returns")
        
        if returns.ndim != 2:
            raise ValueError(f"Returns must be a 2D array, got {returns.ndim}D")
        
        # Verify number of columns matches n_assets
        if returns.shape[1] != self.n_assets:
            raise ValueError(
                f"Number of assets in returns ({returns.shape[1]}) doesn't match model ({self.n_assets})"
            )
        
        # Check for missing values
        if np.isnan(returns).any():
            raise ValueError("Returns contain NaN values")
        
        # Check for infinite values
        if np.isinf(returns).any():
            raise ValueError("Returns contain infinite values")
        
        return returns
    
    def calculate_residuals(
        self,
        returns: np.ndarray,
        covariances: np.ndarray
    ) -> np.ndarray:
        """
        Calculate standardized residuals for the model.
        
        Parameters
        ----------
        returns : np.ndarray
            Multivariate return series with shape (T, N)
        covariances : np.ndarray
            Conditional covariance matrices with shape (T, N, N)
            
        Returns
        -------
        np.ndarray
            Standardized residuals with shape (T, N)
        """
        # Validate input dimensions
        if returns.ndim != 2:
            raise ValueError(f"Returns must be a 2D array, got {returns.ndim}D")
        
        if covariances.ndim != 3:
            raise ValueError(f"Covariances must be a 3D array, got {covariances.ndim}D")
        
        T, N = returns.shape
        
        if covariances.shape[0] != T:
            raise ValueError(f"Number of covariance matrices must match number of returns")
        
        if covariances.shape[1] != N or covariances.shape[2] != N:
            raise ValueError(f"Covariance matrix dimensions must match number of assets")
        
        # Initialize standardized residuals array
        std_residuals = np.zeros_like(returns)
        
        # Calculate standardized residuals using Cholesky decomposition
        for t in range(T):
            try:
                # Cholesky decomposition of covariance matrix
                chol = scipy.linalg.cholesky(covariances[t], lower=True)
                
                # Standardize returns: std_resid = chol^(-1) * returns
                std_residuals[t, :] = scipy.linalg.solve_triangular(
                    chol, returns[t, :], lower=True
                )
            except np.linalg.LinAlgError:
                # Handle non-positive definite matrices
                logger.warning(
                    f"Non-positive definite covariance matrix at time {t}. "
                    "Using eigenvalue regularization."
                )
                
                # Eigenvalue regularization
                eigvals, eigvecs = np.linalg.eigh(covariances[t])
                eigvals = np.maximum(eigvals, 1e-6)  # Ensure positive eigenvalues
                reg_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Recalculate with regularized matrix
                chol = scipy.linalg.cholesky(reg_cov, lower=True)
                std_residuals[t, :] = scipy.linalg.solve_triangular(
                    chol, returns[t, :], lower=True
                )
        
        return std_residuals
    
    def set_parameters(self, params: MultivariateVolatilityParameters) -> None:
        """
        Set model parameters with validation.
        
        Parameters
        ----------
        params : MultivariateVolatilityParameters
            Model parameters
        """
        # Validate parameters match model type
        if self.model_type is not None and params.model_type != self.model_type:
            raise ValueError(
                f"Parameter model type ({params.model_type}) doesn't match model ({self.model_type})"
            )
        
        # Validate n_assets matches parameters dimensions
        n = params.initial_covariance.shape[0]
        if n != self.n_assets:
            raise ValueError(
                f"Parameter dimensions ({n}) don't match model n_assets ({self.n_assets})"
            )
        
        # Validate parameters are valid for this model
        if not params.validate():
            raise ValueError("Parameters failed validation")
        
        # Set parameters
        self.parameters = params
        
        # Log parameter update
        logger.debug(f"Parameters set for {self.model_type} model with {self.n_assets} assets")
    
    @abc.abstractmethod
    def loglikelihood(
        self,
        returns: np.ndarray,
        params: np.ndarray
    ) -> float:
        """
        Calculate model log-likelihood for optimization.
        
        Parameters
        ----------
        returns : np.ndarray
            Multivariate return series
        params : np.ndarray
            Model parameters as a flat vector
            
        Returns
        -------
        float
            Negative log-likelihood value
        """
        pass
    
    @abc.abstractmethod
    def generate_starting_params(
        self,
        returns: np.ndarray
    ) -> MultivariateVolatilityParameters:
        """
        Generate initial parameters.
        
        Parameters
        ----------
        returns : np.ndarray
            Multivariate return series
            
        Returns
        -------
        MultivariateVolatilityParameters
            Initial parameters
        """
        pass
    
    @abc.abstractmethod
    def parameter_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate parameter bounds for optimization.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds
        """
        pass


def register_multivariate_model(model_class: Type[MultivariateVolatilityModel]) -> Type[MultivariateVolatilityModel]:
    """
    Decorator to register a multivariate volatility model class in the global registry.
    
    Parameters
    ----------
    model_class : Type[MultivariateVolatilityModel]
        Model class to register
        
    Returns
    -------
    Type[MultivariateVolatilityModel]
        The registered model class
    """
    # Add the model class to the registry
    MULTIVARIATE_MODELS[model_class.model_type] = model_class
    
    # Log registration
    logger.debug(f"Registered multivariate model type: {model_class.model_type}")
    
    # Return the model class to allow decorator stacking
    return model_class


def create_multivariate_model(
    model_type: str,
    n_assets: int,
    params: Dict[str, Any]
) -> MultivariateVolatilityModel:
    """
    Factory function to create a multivariate volatility model of the specified type.
    
    Parameters
    ----------
    model_type : str
        Name of the multivariate model type to create
    n_assets : int
        Number of assets (dimension of multivariate time series)
    params : Dict[str, Any]
        Dictionary of parameters to pass to the model constructor
        
    Returns
    -------
    MultivariateVolatilityModel
        Instance of the specified multivariate model type
        
    Raises
    ------
    ValueError
        If the specified model type is not registered
    """
    # Check if the model_type exists in the registry
    if model_type not in MULTIVARIATE_MODELS:
        raise ValueError(
            f"Unknown multivariate model type: {model_type}. "
            f"Available types: {list(MULTIVARIATE_MODELS.keys())}"
        )
    
    # Get the model class from the registry
    model_class = MULTIVARIATE_MODELS[model_type]
    
    # Create and return an instance of the model
    model = model_class(n_assets=n_assets, **params)
    
    return model


def estimate_multivariate_volatility(
    returns: np.ndarray,
    model_type: str,
    params: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None
) -> MultivariateVolatilityResult:
    """
    Generic function to estimate parameters for a multivariate volatility model.
    
    Parameters
    ----------
    returns : np.ndarray
        Multivariate return series with shape (T, N)
    model_type : str
        Type of multivariate model to estimate
    params : Optional[Dict[str, Any]], default=None
        Initial parameters for estimation
    options : Optional[Dict[str, Any]], default=None
        Options for estimation
        
    Returns
    -------
    MultivariateVolatilityResult
        Estimation results
    """
    # Validate returns data dimensions
    returns = validate_array(returns, param_name="returns")
    
    if returns.ndim != 2:
        raise ValueError(f"Returns must be a 2D array, got {returns.ndim}D")
    
    # Determine number of assets from returns data
    _, n_assets = returns.shape
    
    # Create a multivariate model of the specified type
    model_params = params or {}
    model = create_multivariate_model(model_type, n_assets, model_params)
    
    # Set up initial parameters if provided
    starting_params = None
    if params and 'starting_params' in params:
        starting_params = params['starting_params']
    
    # Fit the model to the returns data
    result = model.fit(returns, starting_params=starting_params, options=options)
    
    return result


@handle_exceptions_async
async def async_estimate_multivariate_volatility(
    returns: np.ndarray,
    model_type: str,
    params: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None
) -> Coroutine[Any, Any, MultivariateVolatilityResult]:
    """
    Asynchronous version of the multivariate volatility estimation function.
    
    Parameters
    ----------
    returns : np.ndarray
        Multivariate return series with shape (T, N)
    model_type : str
        Type of multivariate model to estimate
    params : Optional[Dict[str, Any]], default=None
        Initial parameters for estimation
    options : Optional[Dict[str, Any]], default=None
        Options for estimation
        
    Returns
    -------
    Coroutine[Any, Any, MultivariateVolatilityResult]
        Coroutine yielding estimation result
    """
    # Validate returns data dimensions
    returns = validate_array(returns, param_name="returns")
    
    if returns.ndim != 2:
        raise ValueError(f"Returns must be a 2D array, got {returns.ndim}D")
    
    # Determine number of assets from returns data
    _, n_assets = returns.shape
    
    # Create a multivariate model of the specified type
    model_params = params or {}
    model = create_multivariate_model(model_type, n_assets, model_params)
    
    # Set up initial parameters if provided
    starting_params = None
    if params and 'starting_params' in params:
        starting_params = params['starting_params']
    
    # Asynchronously fit the model to the returns data
    result = await model.fit_async(returns, starting_params=starting_params, options=options)
    
    return result


def forecast_multivariate_volatility(
    result: MultivariateVolatilityResult,
    horizon: int,
    initial_values: Optional[np.ndarray] = None,
    options: Optional[Dict[str, Any]] = None
) -> MultivariateVolatilityForecast:
    """
    Generate forecasts from a fitted multivariate volatility model.
    
    Parameters
    ----------
    result : MultivariateVolatilityResult
        Result from a fitted multivariate volatility model
    horizon : int
        Forecast horizon
    initial_values : Optional[np.ndarray], default=None
        Initial values for forecasting
    options : Optional[Dict[str, Any]], default=None
        Options for forecasting
        
    Returns
    -------
    MultivariateVolatilityForecast
        Forecast results
    """
    # Validate that result contains a fitted model
    if not isinstance(result, MultivariateVolatilityResult):
        raise TypeError("Result must be a MultivariateVolatilityResult")
    
    # Extract model type and parameters from result
    model_type = result.model_type
    parameters = result.parameters
    n_assets = result.residuals.shape[1]
    
    # Create a new model instance of the same type
    model_params = {}
    model = create_multivariate_model(model_type, n_assets, model_params)
    
    # Initialize the model with fitted parameters
    model.set_parameters(parameters)
    
    # Set up options
    forecast_options = options or {}
    
    # Extract forecast options
    alpha_level = forecast_options.get('alpha_level', None)
    n_simulations = forecast_options.get('n_simulations', None)
    
    # Generate forecasts
    forecast = model.forecast(
        horizon,
        returns=initial_values,
        alpha_level=alpha_level,
        n_simulations=n_simulations
    )
    
    return forecast


def simulate_multivariate_volatility(
    model_type: str,
    params: Dict[str, Any],
    n_obs: int,
    n_assets: int,
    dist_type: Optional[str] = None,
    dist_params: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate multivariate return series with specified volatility dynamics.
    
    Parameters
    ----------
    model_type : str
        Type of multivariate model to simulate
    params : Dict[str, Any]
        Model parameters
    n_obs : int
        Number of observations to simulate
    n_assets : int
        Number of assets (dimension of multivariate time series)
    dist_type : Optional[str], default=None
        Distribution type for innovations
    dist_params : Optional[Dict[str, Any]], default=None
        Distribution parameters
    rng : Optional[np.random.Generator], default=None
        Random number generator
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Simulated returns and conditional covariance matrices
    """
    # Convert distribution type string to enum if provided
    dist_type_enum = None
    if dist_type is not None:
        try:
            dist_type_enum = DistributionType[dist_type.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown distribution type: {dist_type}. "
                f"Available types: {[d.name for d in DistributionType]}"
            )
    
    # Create a multivariate model of the specified type
    model_kwargs = {}
    if dist_type_enum is not None:
        model_kwargs['dist_type'] = dist_type_enum
    if dist_params is not None:
        model_kwargs['dist_params'] = dist_params
    
    model = create_multivariate_model(model_type, n_assets, model_kwargs)
    
    # Create parameters object
    if isinstance(params.get('initial_covariance', None), np.ndarray):
        initial_covariance = params['initial_covariance']
    else:
        # Create identity matrix as default
        initial_covariance = np.eye(n_assets)
    
    model_params = MultivariateVolatilityParameters(
        model_type=model_type,
        model_params={k: v for k, v in params.items() if k != 'initial_covariance'},
        initial_covariance=initial_covariance,
        dist_type=dist_type_enum if dist_type_enum is not None else DistributionType.NORMAL,
        dist_params=dist_params or {}
    )
    
    # Set model parameters
    model.set_parameters(model_params)
    
    # Simulate returns
    returns, covariances = model.simulate(n_obs, rng=rng)
    
    return returns, covariances


@optimized_jit
def calculate_multivariate_loglikelihood(
    returns: np.ndarray,
    covariances: np.ndarray,
    distribution: str,
    dist_params: Dict[str, Any]
) -> float:
    """
    Calculate log-likelihood for multivariate volatility models with various error distributions.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of multivariate returns with shape (T, N)
    covariances : np.ndarray
        Array of conditional covariance matrices with shape (T, N, N)
    distribution : str
        Error distribution type ('NORMAL', 'STUDENT', 'SKEWT')
    dist_params : Dict[str, Any]
        Distribution parameters
        
    Returns
    -------
    float
        Log-likelihood value
    """
    # Validate input arrays and parameters
    if returns.ndim != 2:
        raise ValueError(f"Returns must be a 2D array, got {returns.ndim}D")
    
    if covariances.ndim != 3:
        raise ValueError(f"Covariances must be a 3D array, got {covariances.ndim}D")
    
    T, N = returns.shape
    
    if covariances.shape[0] != T:
        raise ValueError(f"Number of covariance matrices must match number of returns")
    
    if covariances.shape[1] != N or covariances.shape[2] != N:
        raise ValueError(f"Covariance matrix dimensions must match number of assets")
    
    # Initialize log-likelihood
    loglik = 0.0
    
    # Set up appropriate distribution
    if distribution == 'NORMAL':
        # For multivariate normal, we compute:
        # -0.5 * (N * log(2π) + log(det(Σ_t)) + r_t' * Σ_t^(-1) * r_t) for each t
        const_term = -0.5 * N * np.log(2 * np.pi)
        
        for t in range(T):
            # Get current returns and covariance
            r_t = returns[t, :]
            sigma_t = covariances[t, :, :]
            
            try:
                # Compute log determinant of covariance
                chol = scipy.linalg.cholesky(sigma_t, lower=True)
                log_det = 2 * np.sum(np.log(np.diag(chol)))
                
                # Compute quadratic form r_t' * Σ_t^(-1) * r_t
                # First solve Cholesky system: chol * z = r_t
                z = scipy.linalg.solve_triangular(chol, r_t, lower=True)
                
                # Squared Mahalanobis distance is z'z
                quad_form = np.sum(z**2)
                
                # Add to log-likelihood
                loglik += const_term - 0.5 * (log_det + quad_form)
                
            except np.linalg.LinAlgError:
                # Handle non-positive definite matrix
                loglik += -1e6  # Large penalty
        
    elif distribution == 'STUDENT':
        # For multivariate Student's t, we need the degrees of freedom
        df = dist_params.get('df', 8.0)
        
        # Multivariate t log-likelihood
        const_term = scipy.special.gammaln((df + N) / 2) - scipy.special.gammaln(df / 2) \
                    - 0.5 * N * np.log(np.pi * df)
        
        for t in range(T):
            # Get current returns and covariance
            r_t = returns[t, :]
            sigma_t = covariances[t, :, :]
            
            try:
                # Compute log determinant of covariance
                chol = scipy.linalg.cholesky(sigma_t, lower=True)
                log_det = 2 * np.sum(np.log(np.diag(chol)))
                
                # Compute quadratic form r_t' * Σ_t^(-1) * r_t
                z = scipy.linalg.solve_triangular(chol, r_t, lower=True)
                quad_form = np.sum(z**2)
                
                # Compute t log-likelihood
                loglik += const_term - 0.5 * log_det \
                         - 0.5 * (df + N) * np.log(1 + quad_form / df)
                
            except np.linalg.LinAlgError:
                # Handle non-positive definite matrix
                loglik += -1e6  # Large penalty
        
    elif distribution == 'SKEWT':
        # For multivariate skewed t, we use the implementation from SkewedTDistribution
        # This is less numba-friendly and would be implemented in a non-jit version
        loglik = -1e6  # Placeholder - actual implementation would use the SkewedTDistribution class
        
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    
    return loglik


@optimized_jit
def compute_correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix from a covariance matrix.
    
    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix
        
    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    # Validate covariance is symmetric and positive definite
    if not is_symmetric(covariance, raise_error=False):
        raise ValueError("Covariance matrix must be symmetric")
    
    if not is_positive_definite(covariance, raise_error=False):
        raise ValueError("Covariance matrix must be positive definite")
    
    # Extract standard deviations from diagonal elements
    n = covariance.shape[0]
    std_devs = np.sqrt(np.diag(covariance))
    
    # Compute correlation matrix
    correlation = np.zeros_like(covariance)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                correlation[i, j] = 1.0
            else:
                correlation[i, j] = covariance[i, j] / (std_devs[i] * std_devs[j])
    
    # Ensure numerical stability (diagonal elements must be 1.0)
    np.fill_diagonal(correlation, 1.0)
    
    return correlation


@optimized_jit
def compute_covariance_from_correlation(correlation: np.ndarray, std_devs: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix from correlation matrix and standard deviations.
    
    Parameters
    ----------
    correlation : np.ndarray
        Correlation matrix
    std_devs : np.ndarray
        Vector of standard deviations
        
    Returns
    -------
    np.ndarray
        Covariance matrix
    """
    # Validate correlation is symmetric with diagonal elements of 1.0
    if not is_symmetric(correlation, raise_error=False):
        raise ValueError("Correlation matrix must be symmetric")
    
    # Check if diagonal elements are approximately 1.0
    if not np.allclose(np.diag(correlation), 1.0):
        raise ValueError("Diagonal elements of correlation matrix must be 1.0")
    
    # Validate std_devs are positive
    if np.any(std_devs <= 0):
        raise ValueError("Standard deviations must be positive")
    
    # Compute covariance matrix
    n = correlation.shape[0]
    covariance = np.zeros_like(correlation)
    
    for i in range(n):
        for j in range(n):
            covariance[i, j] = correlation[i, j] * std_devs[i] * std_devs[j]
    
    return covariance