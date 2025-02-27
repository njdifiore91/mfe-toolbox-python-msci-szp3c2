"""
MFE Toolbox Example - Creating and Using Custom Distributions

This example demonstrates how to extend the MFE Toolbox's distribution framework by
implementing a custom statistical distribution - the Asymmetric Laplace Distribution.
The implementation showcases Numba optimization, parameter estimation via maximum
likelihood, and comparison with standard distributions.

The Asymmetric Laplace Distribution is useful for financial modeling due to its
ability to capture both skewness and fat tails often observed in financial returns.
"""

import numpy as np  # numpy 1.26.3
import scipy.special  # scipy 1.11.4
import scipy.optimize  # scipy 1.11.4
import matplotlib.pyplot as plt  # matplotlib 3.7.2
from typing import Dict, List, Optional, Tuple, Union, Any  # Python 3.12
import numba  # numba 0.59.0

# Import core distribution classes for comparison
from mfe.core.distributions import GeneralizedErrorDistribution, SkewedTDistribution, DistributionTest
from mfe.utils.numba_helpers import optimized_jit
from mfe.utils.validation import validate_type, is_positive_float
from mfe.utils.numpy_helpers import ensure_array


@optimized_jit()
def asymmetric_laplace_pdf(x: np.ndarray, mu: float, sigma: float, kappa: float) -> np.ndarray:
    """
    Probability density function for the Asymmetric Laplace Distribution, optimized with Numba.
    
    Parameters
    ----------
    x : ndarray
        Points at which to evaluate the PDF
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    kappa : float
        Asymmetry parameter (must be positive)
        
    Returns
    -------
    ndarray
        Asymmetric Laplace PDF values for given inputs
    """
    # Calculate normalization factor
    norm_factor = 2.0 / (sigma * (kappa + 1.0/kappa))
    
    # Initialize output array
    pdf = np.zeros_like(x, dtype=np.float64)
    
    # Calculate PDF values for each point
    for i in range(len(x)):
        # Compute standardized difference
        z = x[i] - mu
        
        if z < 0:
            # Left tail formula
            pdf[i] = norm_factor * kappa * np.exp(kappa * z / sigma)
        else:
            # Right tail formula
            pdf[i] = norm_factor * (1.0/kappa) * np.exp(-z / (kappa * sigma))
    
    return pdf


@optimized_jit()
def asymmetric_laplace_cdf(x: np.ndarray, mu: float, sigma: float, kappa: float) -> np.ndarray:
    """
    Cumulative distribution function for the Asymmetric Laplace Distribution, optimized with Numba.
    
    Parameters
    ----------
    x : ndarray
        Points at which to evaluate the CDF
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    kappa : float
        Asymmetry parameter (must be positive)
        
    Returns
    -------
    ndarray
        Asymmetric Laplace CDF values for given inputs
    """
    # Initialize output array
    cdf = np.zeros_like(x, dtype=np.float64)
    
    # Calculate CDF values for each point
    for i in range(len(x)):
        # Compute standardized difference
        z = x[i] - mu
        
        if z < 0:
            # Left tail formula
            cdf[i] = (kappa**2 / (1 + kappa**2)) * np.exp(kappa * z / sigma)
        else:
            # Right tail formula
            cdf[i] = 1 - (1 / (1 + kappa**2)) * np.exp(-z / (kappa * sigma))
    
    return cdf


def asymmetric_laplace_ppf(q: np.ndarray, mu: float, sigma: float, kappa: float) -> np.ndarray:
    """
    Percent point function (inverse CDF) for the Asymmetric Laplace Distribution.
    
    Parameters
    ----------
    q : ndarray
        Probability points at which to evaluate the PPF
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    kappa : float
        Asymmetry parameter (must be positive)
        
    Returns
    -------
    ndarray
        Asymmetric Laplace quantile values for given probability inputs
    """
    # Validate inputs
    q = ensure_array(q)
    if not np.all((q >= 0) & (q <= 1)):
        raise ValueError("All probabilities must be between 0 and 1")
    
    # Critical probability value for the Asymmetric Laplace Distribution
    critical_prob = kappa**2 / (1 + kappa**2)
    
    # Initialize output array
    result = np.empty_like(q, dtype=np.float64)
    
    # Calculate quantiles for each probability
    for i in range(len(q.flat)):
        qi = q.flat[i]
        
        # Handle boundary cases
        if qi == 0:
            result.flat[i] = -np.inf
        elif qi == 1:
            result.flat[i] = np.inf
        else:
            # Left tail formula (q <= critical_prob)
            if qi <= critical_prob:
                result.flat[i] = mu + (sigma / kappa) * np.log(qi / critical_prob)
            # Right tail formula (q > critical_prob)
            else:
                result.flat[i] = mu - sigma * kappa * np.log((1 - qi) / (1 - critical_prob))
    
    return result


def asymmetric_laplace_random(size: int, mu: float, sigma: float, kappa: float, 
                             seed: Optional[int] = None) -> np.ndarray:
    """
    Random number generator for the Asymmetric Laplace Distribution.
    
    Parameters
    ----------
    size : int
        Number of random samples to generate
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    kappa : float
        Asymmetry parameter (must be positive)
    seed : Optional[int], default=None
        Random seed for reproducibility
        
    Returns
    -------
    ndarray
        Array of random numbers from the Asymmetric Laplace distribution
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate uniform random numbers
    u = np.random.random(size)
    
    # Transform to Asymmetric Laplace using inverse CDF
    return asymmetric_laplace_ppf(u, mu, sigma, kappa)


def estimate_asymmetric_laplace_params(data: np.ndarray, 
                                      starting_values: Optional[Dict[str, float]] = None,
                                      bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
    """
    Estimates parameters of the Asymmetric Laplace Distribution from data using maximum likelihood.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    starting_values : Optional[dict], default=None
        Initial parameter values for optimization. If None, defaults are used.
    bounds : Optional[dict], default=None
        Parameter bounds for optimization. If None, defaults are used.
        
    Returns
    -------
    dict
        Estimated parameters (mu, sigma, kappa) and optimization results
    """
    # Validate and prepare data
    data = ensure_array(data)
    
    # Set default starting values if not provided
    if starting_values is None:
        starting_values = {
            'mu': np.median(data),  # More robust than mean for asymmetric distributions
            'sigma': np.std(data, ddof=1),
            'kappa': np.sqrt(np.sum(data > np.median(data)) / np.sum(data < np.median(data)))
        }
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'mu': (np.min(data) - 5 * np.std(data), np.max(data) + 5 * np.std(data)),
            'sigma': (1e-6, 10 * np.std(data)),
            'kappa': (0.01, 100.0)  # Allow a wide range of asymmetry
        }
    
    # Define negative log-likelihood function
    def neg_loglikelihood(params):
        mu, sigma, kappa = params
        if sigma <= 0 or kappa <= 0:
            return 1e10  # Large penalty for invalid parameters
        
        # Compute PDF values
        pdf_values = asymmetric_laplace_pdf(data, mu, sigma, kappa)
        
        # Avoid log(0) by adding a small constant
        eps = np.finfo(float).eps
        log_likelihood = np.sum(np.log(pdf_values + eps))
        
        return -log_likelihood
    
    # Initial parameter vector
    x0 = [starting_values['mu'], starting_values['sigma'], starting_values['kappa']]
    
    # Parameter bounds for optimization
    param_bounds = [
        bounds['mu'],
        bounds['sigma'],
        bounds['kappa']
    ]
    
    # Perform optimization
    result = scipy.optimize.minimize(
        neg_loglikelihood,
        x0,
        bounds=param_bounds,
        method='L-BFGS-B'
    )
    
    # Extract estimated parameters
    mu_hat, sigma_hat, kappa_hat = result.x
    
    # Calculate log-likelihood
    ll = -result.fun
    
    # Calculate information criteria
    n = len(data)
    k = 3  # Number of parameters
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)
    
    # Return results
    return {
        'mu': mu_hat,
        'sigma': sigma_hat,
        'kappa': kappa_hat,
        'loglikelihood': ll,
        'aic': aic,
        'bic': bic,
        'converged': result.success,
        'message': result.message
    }


class AsymmetricLaplaceDistribution:
    """
    Class implementation of the Asymmetric Laplace Distribution with methods for 
    PDF, CDF, PPF, and random sampling.
    """
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0, kappa: float = 1.0):
        """
        Initialize an Asymmetric Laplace distribution with location, scale, and asymmetry parameters.
        
        Parameters
        ----------
        mu : float, default=0.0
            Location parameter
        sigma : float, default=1.0
            Scale parameter (must be positive)
        kappa : float, default=1.0
            Asymmetry parameter (must be positive)
        """
        # Validate parameters
        validate_type(mu, float, "mu")
        is_positive_float(sigma, False, "sigma")
        is_positive_float(kappa, False, "kappa")
        
        self.mu = mu
        self.sigma = sigma
        self.kappa = kappa
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability density function for the Asymmetric Laplace distribution.
        
        Parameters
        ----------
        x : Union[float, ndarray]
            Points at which to evaluate the PDF
            
        Returns
        -------
        Union[float, ndarray]
            PDF values for given inputs
        """
        x_array = ensure_array(x)
        return asymmetric_laplace_pdf(x_array, self.mu, self.sigma, self.kappa)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function for the Asymmetric Laplace distribution.
        
        Parameters
        ----------
        x : Union[float, ndarray]
            Points at which to evaluate the CDF
            
        Returns
        -------
        Union[float, ndarray]
            CDF values for given inputs
        """
        x_array = ensure_array(x)
        return asymmetric_laplace_cdf(x_array, self.mu, self.sigma, self.kappa)
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Percent point function (inverse CDF) for the Asymmetric Laplace distribution.
        
        Parameters
        ----------
        q : Union[float, ndarray]
            Probability points at which to evaluate the PPF
            
        Returns
        -------
        Union[float, ndarray]
            Quantile values for given probabilities
        """
        q_array = ensure_array(q)
        return asymmetric_laplace_ppf(q_array, self.mu, self.sigma, self.kappa)
    
    def random(self, size: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the Asymmetric Laplace distribution.
        
        Parameters
        ----------
        size : int
            Number of random samples to generate
        seed : Optional[int], default=None
            Random seed for reproducibility
            
        Returns
        -------
        ndarray
            Array of random samples
        """
        return asymmetric_laplace_random(size, self.mu, self.sigma, self.kappa, seed)
    
    @classmethod
    def fit(cls, data: np.ndarray, 
            starting_values: Optional[Dict[str, float]] = None,
            bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> 'AsymmetricLaplaceDistribution':
        """
        Fit distribution parameters to data using maximum likelihood estimation.
        
        Parameters
        ----------
        data : ndarray
            Input data array
        starting_values : Optional[dict], default=None
            Initial parameter values for optimization. If None, defaults are used.
        bounds : Optional[dict], default=None
            Parameter bounds for optimization. If None, defaults are used.
            
        Returns
        -------
        AsymmetricLaplaceDistribution
            New instance with fitted parameters
        """
        params = estimate_asymmetric_laplace_params(data, starting_values, bounds)
        return cls(params['mu'], params['sigma'], params['kappa'])
    
    def loglikelihood(self, data: np.ndarray) -> float:
        """
        Calculate log-likelihood of data under the Asymmetric Laplace distribution.
        
        Parameters
        ----------
        data : ndarray
            Input data array
            
        Returns
        -------
        float
            Log-likelihood value
        """
        data = ensure_array(data)
        pdf_values = self.pdf(data)
        
        # Avoid log(0) by adding a small constant
        eps = np.finfo(float).eps
        return np.sum(np.log(pdf_values + eps))


def plot_distribution_comparison(data: np.ndarray, 
                               show_pdf: bool = True,
                               show_cdf: bool = True, 
                               show_qq: bool = False) -> Tuple:
    """
    Compares standard distributions with the custom Asymmetric Laplace distribution.
    
    Parameters
    ----------
    data : ndarray
        Data to fit distributions to
    show_pdf : bool, default=True
        Whether to show PDF comparison plot
    show_cdf : bool, default=True
        Whether to show CDF comparison plot
    show_qq : bool, default=False
        Whether to show Q-Q plot comparison
        
    Returns
    -------
    tuple
        Matplotlib figure and axes objects
    """
    # Determine number of subplots
    num_plots = sum([show_pdf, show_cdf, show_qq])
    if num_plots == 0:
        raise ValueError("At least one plot type must be selected")
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]  # Ensure axes is a list for consistent indexing
    
    # Fit standard distributions
    ged = GeneralizedErrorDistribution.fit(data)
    skewt = SkewedTDistribution.fit(data)
    
    # Fit custom Asymmetric Laplace distribution
    ald = AsymmetricLaplaceDistribution.fit(data)
    
    # Define common x-range for plots
    x_min = np.min(data) - 2 * np.std(data)
    x_max = np.max(data) + 2 * np.std(data)
    x = np.linspace(x_min, x_max, 1000)
    
    # Define colors for different distributions
    colors = {
        'GED': 'blue',
        'Skewed-t': 'red',
        'Asymmetric Laplace': 'green',
        'Data': 'gray'
    }
    
    plot_idx = 0
    
    # Plot PDF comparison
    if show_pdf:
        ax = axes[plot_idx]
        
        # Plot histogram of empirical data
        ax.hist(data, bins=30, density=True, alpha=0.5, color=colors['Data'], label='Data')
        
        # Plot fitted PDFs
        ax.plot(x, ged.pdf(x), label=f'GED (ν={ged.nu:.2f})', color=colors['GED'])
        ax.plot(x, skewt.pdf(x), label=f'Skewed-t (ν={skewt.nu:.2f}, λ={skewt.lambda_:.2f})', 
                color=colors['Skewed-t'])
        ax.plot(x, ald.pdf(x), label=f'Asymmetric Laplace (κ={ald.kappa:.2f})', 
                color=colors['Asymmetric Laplace'])
        
        ax.set_title('PDF Comparison')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        plot_idx += 1
    
    # Plot CDF comparison
    if show_cdf:
        ax = axes[plot_idx]
        
        # Plot empirical CDF
        from statsmodels.distributions import ECDF
        ecdf = ECDF(data)
        ax.step(np.sort(data), ecdf(np.sort(data)), label='Empirical', color=colors['Data'])
        
        # Plot fitted CDFs
        ax.plot(x, ged.cdf(x), label=f'GED (ν={ged.nu:.2f})', color=colors['GED'])
        ax.plot(x, skewt.cdf(x), label=f'Skewed-t (ν={skewt.nu:.2f}, λ={skewt.lambda_:.2f})', 
                color=colors['Skewed-t'])
        ax.plot(x, ald.cdf(x), label=f'Asymmetric Laplace (κ={ald.kappa:.2f})', 
                color=colors['Asymmetric Laplace'])
        
        ax.set_title('CDF Comparison')
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability')
        ax.legend()
        plot_idx += 1
    
    # Plot QQ comparison if requested
    if show_qq:
        ax = axes[plot_idx]
        
        # Create reference line
        min_q = np.min(data)
        max_q = np.max(data)
        ref_line = np.linspace(min_q, max_q, 100)
        ax.plot(ref_line, ref_line, 'k--', label='Reference Line')
        
        # Get empirical quantiles
        p = np.linspace(0.01, 0.99, 100)
        q_emp = np.quantile(data, p)
        
        # Plot QQ plots for each distribution
        q_ged = ged.ppf(p)
        q_skewt = skewt.ppf(p)
        q_ald = ald.ppf(p)
        
        ax.scatter(q_ged, q_emp, label=f'GED (ν={ged.nu:.2f})', 
                  color=colors['GED'], alpha=0.7)
        ax.scatter(q_skewt, q_emp, label=f'Skewed-t (ν={skewt.nu:.2f}, λ={skewt.lambda_:.2f})', 
                  color=colors['Skewed-t'], alpha=0.7)
        ax.scatter(q_ald, q_emp, label=f'Asymmetric Laplace (κ={ald.kappa:.2f})', 
                  color=colors['Asymmetric Laplace'], alpha=0.7)
        
        ax.set_title('Q-Q Plot Comparison')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Empirical Quantiles')
        ax.legend()
    
    plt.tight_layout()
    return fig, axes


def generate_financial_example(n_samples: int = 1000, 
                             mu: float = 0.0005, 
                             sigma: float = 0.01, 
                             kappa: float = 1.5,
                             seed: Optional[int] = 42) -> np.ndarray:
    """
    Generates a simulated financial return series for demonstrating distribution fitting.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
    mu : float, default=0.0005
        Location parameter (daily mean return)
    sigma : float, default=0.01
        Scale parameter (base volatility)
    kappa : float, default=1.5
        Asymmetry parameter (negative skew when > 1)
    seed : Optional[int], default=42
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Simulated financial return series
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Generate base returns from Asymmetric Laplace distribution
    returns = asymmetric_laplace_random(n_samples, mu, sigma, kappa)
    
    # Add some volatility clustering (GARCH-like behavior)
    vol_multiplier = np.ones(n_samples)
    rho = 0.94  # Persistence in volatility
    
    for i in range(1, n_samples):
        # Volatility depends on previous volatility and squared return
        vol_multiplier[i] = np.sqrt(0.05 + 0.85 * vol_multiplier[i-1]**2 + 0.1 * returns[i-1]**2)
    
    # Apply volatility multiplier to returns
    returns = returns * vol_multiplier
    
    return returns


def demo_distribution_fit():
    """
    Demonstrates fitting different distributions to simulated data and evaluating the fit.
    """
    # Generate simulated financial data
    print("Generating simulated financial returns with Asymmetric Laplace characteristics...")
    returns = generate_financial_example(2000)
    
    print(f"Data statistics:")
    print(f"  Mean: {np.mean(returns):.6f}")
    print(f"  Std Dev: {np.std(returns):.6f}")
    print(f"  Skewness: {scipy.stats.skew(returns):.6f}")
    print(f"  Kurtosis: {scipy.stats.kurtosis(returns):.6f}")
    print("")
    
    # Use DistributionTest to compare distributions
    dist_test = DistributionTest(returns)
    comparison = dist_test.compare_distributions(['norm', 'ged', 'skewt'])
    
    # Add our custom distribution
    ald = AsymmetricLaplaceDistribution.fit(returns)
    ald_ll = ald.loglikelihood(returns)
    
    # Calculate information criteria for our custom distribution
    n = len(returns)
    k = 3  # Number of parameters
    ald_aic = -2 * ald_ll + 2 * k
    ald_bic = -2 * ald_ll + k * np.log(n)
    
    # Print comparison results
    print("Model comparison:")
    print(f"  Normal: AIC = {comparison['aic_values']['norm']:.2f}, BIC = {comparison['bic_values']['norm']:.2f}")
    print(f"  GED: AIC = {comparison['aic_values']['ged']:.2f}, BIC = {comparison['bic_values']['ged']:.2f}")
    print(f"  Skewed-t: AIC = {comparison['aic_values']['skewt']:.2f}, BIC = {comparison['bic_values']['skewt']:.2f}")
    print(f"  Asymmetric Laplace: AIC = {ald_aic:.2f}, BIC = {ald_bic:.2f}")
    print("")
    
    # Print best model based on AIC
    aic_values = comparison['aic_values'].copy()
    aic_values['ald'] = ald_aic
    best_aic = min(aic_values, key=aic_values.get)
    
    # Print best model based on BIC
    bic_values = comparison['bic_values'].copy()
    bic_values['ald'] = ald_bic
    best_bic = min(bic_values, key=bic_values.get)
    
    print(f"Best model by AIC: {best_aic}")
    print(f"Best model by BIC: {best_bic}")
    print("")
    
    # Print fitted parameters for Asymmetric Laplace Distribution
    print("Fitted Asymmetric Laplace Distribution parameters:")
    print(f"  mu (location): {ald.mu:.6f}")
    print(f"  sigma (scale): {ald.sigma:.6f}")
    print(f"  kappa (asymmetry): {ald.kappa:.6f}")
    print("")
    
    # Create comparison plots
    print("Creating distribution comparison plots...")
    fig, axes = plot_distribution_comparison(returns, show_pdf=True, show_cdf=True)
    plt.show()
    
    print("Example complete.")


if __name__ == "__main__":
    # Run the demonstration
    demo_distribution_fit()