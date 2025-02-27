"""
Example demonstrating how to create and use custom probability distributions in the MFE Toolbox.

This file showcases how to implement a custom distribution with PDF, CDF, PPF, and random
sampling methods, extending the core distributions module functionality with Numba optimization.
"""
import numpy as np  # numpy 1.26.3
import scipy.special  # scipy 1.11.4
import scipy.stats  # scipy 1.11.4
import matplotlib.pyplot as plt  # matplotlib 3.8.2
from typing import Union, Optional, Dict, Tuple  # Python 3.12
import numba  # numba 0.59.0

from mfe.core.distributions import GeneralizedErrorDistribution, SkewedTDistribution, distribution_fit, jarque_bera, ks_test  # internal import
from mfe.utils.numba_helpers import optimized_jit  # internal import
from mfe.utils.validation import validate_array  # internal import
from mfe.utils.numpy_helpers import ensure_array  # internal import

# Define a small constant to avoid division by zero
EPSILON = 1e-10

@optimized_jit()
def custom_pdf(
    x: np.ndarray, mu1: float, sigma1: float, mu2: float, sigma2: float, weight: float
) -> np.ndarray:
    """
    Probability density function for the custom mixture distribution, optimized with Numba.

    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate the PDF
    mu1 : float
        Location parameter for the first normal distribution
    sigma1 : float
        Scale parameter for the first normal distribution
    mu2 : float
        Location parameter for the second normal distribution
    sigma2 : float
        Scale parameter for the second normal distribution
    weight : float
        Weighting factor for the mixture (0 to 1)

    Returns
    -------
    np.ndarray
        Custom distribution probability density values for given inputs
    """
    # Validate input parameters
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x must be a NumPy array.")
    if not all(isinstance(param, float) for param in [mu1, sigma1, mu2, sigma2, weight]):
        raise TypeError("Distribution parameters must be floats.")

    # Calculate normal PDF for first component using mu1, sigma1
    pdf1 = (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)

    # Calculate normal PDF for second component using mu2, sigma2
    pdf2 = (1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)

    # Compute weighted sum using weight parameter
    pdf = weight * pdf1 + (1 - weight) * pdf2

    # Return resulting PDF values
    return pdf


@optimized_jit()
def custom_cdf(
    x: np.ndarray, mu1: float, sigma1: float, mu2: float, sigma2: float, weight: float
) -> np.ndarray:
    """
    Cumulative distribution function for the custom mixture distribution, optimized with Numba.

    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate the CDF
    mu1 : float
        Location parameter for the first normal distribution
    sigma1 : float
        Scale parameter for the first normal distribution
    mu2 : float
        Location parameter for the second normal distribution
    sigma2 : float
        Scale parameter for the second normal distribution
    weight : float
        Weighting factor for the mixture (0 to 1)

    Returns
    -------
    np.ndarray
        Custom distribution cumulative probability values for given inputs
    """
    # Validate input parameters
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x must be a NumPy array.")
    if not all(isinstance(param, float) for param in [mu1, sigma1, mu2, sigma2, weight]):
        raise TypeError("Distribution parameters must be floats.")

    # Calculate normal CDF for first component using mu1, sigma1
    cdf1 = 0.5 * (1 + scipy.special.erf((x - mu1) / (sigma1 * np.sqrt(2))))

    # Calculate normal CDF for second component using mu2, sigma2
    cdf2 = 0.5 * (1 + scipy.special.erf((x - mu2) / (sigma2 * np.sqrt(2))))

    # Compute weighted sum using weight parameter
    cdf = weight * cdf1 + (1 - weight) * cdf2

    # Return resulting CDF values
    return cdf


def custom_ppf(
    q: np.ndarray, mu1: float, sigma1: float, mu2: float, sigma2: float, weight: float
) -> np.ndarray:
    """
    Percent point function (inverse CDF) for the custom mixture distribution.

    Parameters
    ----------
    q : np.ndarray
        Probabilities for which to calculate the PPF
    mu1 : float
        Location parameter for the first normal distribution
    sigma1 : float
        Scale parameter for the first normal distribution
    mu2 : float
        Location parameter for the second normal distribution
    sigma2 : float
        Scale parameter for the second normal distribution
    weight : float
        Weighting factor for the mixture (0 to 1)

    Returns
    -------
    np.ndarray
        Custom distribution quantile values for given probability inputs
    """
    # Validate input parameters
    q = validate_array(q)
    if not all(isinstance(param, float) for param in [mu1, sigma1, mu2, sigma2, weight]):
        raise TypeError("Distribution parameters must be floats.")

    # Check that probabilities are between 0 and 1
    if not np.all((q >= 0) & (q <= 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    # Implement numerical inversion of CDF for mixture distribution
    def objective(x, target_q):
        return custom_cdf(x, mu1, sigma1, mu2, sigma2, weight) - target_q

    # Use root-finding to compute quantiles
    quantiles = np.zeros_like(q)
    for i, prob in enumerate(q):
        # Initial guess using a simple average
        initial_guess = mu1 * weight + mu2 * (1 - weight)
        result = scipy.optimize.fsolve(objective, initial_guess, args=(prob,))
        quantiles[i] = result[0]

    # Return quantile values
    return quantiles


def custom_random(
    size: int,
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
    weight: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Random number generator for the custom mixture distribution.

    Parameters
    ----------
    size : int
        Number of random samples to generate
    mu1 : float
        Location parameter for the first normal distribution
    sigma1 : float
        Scale parameter for the first normal distribution
    mu2 : float
        Location parameter for the second normal distribution
    sigma2 : float
        Scale parameter for the second normal distribution
    weight : float
        Weighting factor for the mixture (0 to 1)
    seed : Optional[int], default=None
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of random numbers from the custom mixture distribution
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Generate uniform random numbers for component selection
    u = np.random.uniform(size=size)

    # Generate normal random numbers from first component distribution
    samples1 = np.random.normal(mu1, sigma1, size=size)

    # Generate normal random numbers from second component distribution
    samples2 = np.random.normal(mu2, sigma2, size=size)

    # Combine samples based on weight parameter
    samples = np.where(u < weight, samples1, samples2)

    # Return random values from the custom mixture distribution
    return samples


class MixtureDistribution:
    """
    Class implementation of a custom mixture distribution with methods for PDF, CDF, PPF, and random sampling.
    """

    def __init__(self, mu1: float, sigma1: float, mu2: float, sigma2: float, weight: float):
        """
        Initialize a custom mixture distribution with location, scale, and mixture weight parameters.

        Parameters
        ----------
        mu1 : float
            Location parameter for the first normal distribution
        sigma1 : float
            Scale parameter for the first normal distribution
        mu2 : float
            Location parameter for the second normal distribution
        sigma2 : float
            Scale parameter for the second normal distribution
        weight : float
             Weighting factor for the mixture (0 to 1)
        """
        # Validate input parameters
        if not all(isinstance(param, float) for param in [mu1, sigma1, mu2, sigma2, weight]):
            raise TypeError("Distribution parameters must be floats.")

        # Check that sigma1 > 0 and sigma2 > 0
        if sigma1 <= 0 or sigma2 <= 0:
            raise ValueError("Scale parameters (sigma1, sigma2) must be positive.")

        # Check that weight is between 0 and 1
        if not 0 <= weight <= 1:
            raise ValueError("Weight parameter must be between 0 and 1.")

        # Store parameters as instance variables
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.weight = weight

        # Initialize distribution
        self._initialize_distribution()

    def _initialize_distribution(self):
        """
        Initializes the distribution (currently a placeholder).
        """
        pass

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability density function for the custom mixture distribution.

        Parameters
        ----------
        x : Union[float, np.ndarray]
            Points at which to evaluate the PDF

        Returns
        -------
        Union[float, np.ndarray]
            PDF values for given inputs
        """
        # Call custom_pdf function with instance parameters
        return custom_pdf(x, self.mu1, self.sigma1, self.mu2, self.sigma2, self.weight)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function for the custom mixture distribution.

        Parameters
        ----------
        x : Union[float, np.ndarray]
            Points at which to evaluate the CDF

        Returns
        -------
        Union[float, np.ndarray]
            CDF values for given inputs
        """
        # Call custom_cdf function with instance parameters
        return custom_cdf(x, self.mu1, self.sigma1, self.mu2, self.sigma2, self.weight)

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Percent point function (inverse CDF) for the custom mixture distribution.

        Parameters
        ----------
        q : Union[float, np.ndarray]
            Probabilities for which to calculate the PPF

        Returns
        -------
        Union[float, np.ndarray]
            Quantile values for given probabilities
        """
        # Call custom_ppf function with instance parameters
        return custom_ppf(q, self.mu1, self.sigma1, self.mu2, self.sigma2, self.weight)

    def random(self, size: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the custom mixture distribution.

        Parameters
        ----------
        size : int
            Number of random samples to generate
        seed : Optional[int], default=None
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Array of random samples
        """
        # Call custom_random function with instance parameters
        return custom_random(size, self.mu1, self.sigma1, self.mu2, self.sigma2, self.weight, seed)

    def fit(self, data: np.ndarray, starting_values: Optional[Dict] = None, bounds: Optional[Dict] = None) -> "MixtureDistribution":
        """
        Fit distribution parameters to data using maximum likelihood estimation.

        Parameters
        ----------
        data : np.ndarray
            Input data array
        starting_values : Optional[dict], default=None
            Initial parameter values for optimization. If None, defaults are used.
        bounds : Optional[dict], default=None
            Parameter bounds for optimization. If None, defaults are used.

        Returns
        -------
        MixtureDistribution
            New instance with fitted parameters
        """
        # Call estimate_custom_params function
        estimated_params = estimate_custom_params(data, starting_values, bounds)

        # Create new instance with estimated parameters
        new_instance = MixtureDistribution(
            estimated_params["mu1"],
            estimated_params["sigma1"],
            estimated_params["mu2"],
            estimated_params["sigma2"],
            estimated_params["weight"],
        )

        # Return the new instance
        return new_instance

    def loglikelihood(self, data: np.ndarray) -> float:
        """
        Calculate log-likelihood of data under the custom mixture distribution.

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        float
            Log-likelihood value
        """
        # Calculate PDF for each data point using pdf method
        pdf_values = self.pdf(data)

        # Take natural logarithm of PDF values
        log_pdf_values = np.log(pdf_values + EPSILON)  # Adding a small constant to avoid log(0)

        # Sum log-likelihood values
        total_log_likelihood = np.sum(log_pdf_values)

        # Return total log-likelihood
        return total_log_likelihood

    def plot(
        self, x_range: np.ndarray, show_pdf: bool = True, show_cdf: bool = True, show_components: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the PDF and/or CDF of the custom mixture distribution.

        Parameters
        ----------
        x_range : np.ndarray
            X-values for plotting
        show_pdf : bool, default=True
            Whether to plot the PDF
        show_cdf : bool, default=True
            Whether to plot the CDF
        show_components : bool, default=False
            Whether to plot the individual mixture components

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects from matplotlib
        """
        # Create matplotlib figure with required subplots
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()  # Create a second axes that shares the same x-axis

        # Calculate distribution values over x_range
        x = ensure_array(x_range)
        pdf_values = self.pdf(x)
        cdf_values = self.cdf(x)

        # Plot PDF if show_pdf is True
        if show_pdf:
            ax1.plot(x, pdf_values, label="Mixture PDF", color="blue")
            ax1.set_ylabel("PDF", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")

        # Plot CDF if show_cdf is True
        if show_cdf:
            ax2.plot(x, cdf_values, label="Mixture CDF", color="red")
            ax2.set_ylabel("CDF", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

        # If show_components is True, plot individual mixture components
        if show_components:
            pdf1 = (1 / (self.sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mu1) / self.sigma1) ** 2)
            pdf2 = (1 / (self.sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mu2) / self.sigma2) ** 2)
            ax1.plot(x, pdf1, label="Component 1", linestyle="--", color="green")
            ax1.plot(x, pdf2, label="Component 2", linestyle="--", color="orange")

        # Add legends, labels, and titles
        ax1.set_xlabel("X")
        fig.suptitle("Mixture Distribution")
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

        # Return figure and axes for further customization
        return fig, (ax1, ax2)


def estimate_custom_params(
    data: np.ndarray, starting_values: Optional[Dict] = None, bounds: Optional[Dict] = None
) -> Dict:
    """
    Estimate parameters of the custom distribution from data using maximum likelihood.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    starting_values : Optional[dict], default=None
        Initial parameter values for optimization. If None, defaults are used.
    bounds : Optional[dict], default=None
        Parameter bounds for optimization. If None, defaults are used.

    Returns
    -------
    dict
        Estimated parameters and optimization results
    """
    # Validate input data
    data = validate_array(data)

    # Set default starting values if not provided
    if starting_values is None:
        starting_values = {
            "mu1": np.mean(data) - 1,
            "sigma1": np.std(data),
            "mu2": np.mean(data) + 1,
            "sigma2": np.std(data),
            "weight": 0.5,
        }

    # Define objective function as negative log-likelihood using custom_pdf
    def neg_loglikelihood(params):
        mu1, sigma1, mu2, sigma2, weight = params
        if sigma1 <= 0 or sigma2 <= 0 or not 0 <= weight <= 1:
            return np.inf  # Return a large value for invalid parameters
        pdf_values = custom_pdf(data, mu1, sigma1, mu2, sigma2, weight)
        log_likelihood = np.sum(np.log(pdf_values + EPSILON))  # Adding a small constant to avoid log(0)
        return -log_likelihood

    # Define parameter constraints for valid distribution
    if bounds is None:
        bounds = [
            (np.min(data) - 3 * np.std(data), np.max(data) + 3 * np.std(data)),  # mu1
            (EPSILON, 3 * np.std(data)),  # sigma1
            (np.min(data) - 3 * np.std(data), np.max(data) + 3 * np.std(data)),  # mu2
            (EPSILON, 3 * np.std(data)),  # sigma2
            (0, 1),  # weight
        ]

    # Perform optimization using SciPy's optimize.minimize
    initial_params = [
        starting_values["mu1"],
        starting_values["sigma1"],
        starting_values["mu2"],
        starting_values["sigma2"],
        starting_values["weight"],
    ]
    result = scipy.optimize.minimize(neg_loglikelihood, initial_params, bounds=bounds)

    # Extract estimated parameters
    mu1_hat, sigma1_hat, mu2_hat, sigma2_hat, weight_hat = result.x

    # Return estimated parameters and optimization results
    return {
        "mu1": mu1_hat,
        "sigma1": sigma1_hat,
        "mu2": mu2_hat,
        "sigma2": sigma2_hat,
        "weight": weight_hat,
        "success": result.success,
        "message": result.message,
    }


def plot_distribution_comparison(
    custom_params: Dict, x_range: np.ndarray, show_pdf: bool = True, show_cdf: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot custom distribution alongside standard distributions for comparison.

    Parameters
    ----------
    custom_params : dict
        Parameters for the custom mixture distribution
    x_range : np.ndarray
        X-values for plotting
    show_pdf : bool, default=True
        Whether to plot the PDF
    show_cdf : bool, default=True
        Whether to plot the CDF

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects from matplotlib
    """
    # Create matplotlib figure with required subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axes

    # Calculate custom distribution PDF/CDF values over x_range
    x = ensure_array(x_range)
    pdf_custom = custom_pdf(
        x, custom_params["mu1"], custom_params["sigma1"], custom_params["mu2"], custom_params["sigma2"], custom_params["weight"]
    )
    cdf_custom = custom_cdf(
        x, custom_params["mu1"], custom_params["sigma1"], custom_params["mu2"], custom_params["sigma2"], custom_params["weight"]
    )

    # Calculate standard normal distribution values for comparison
    pdf_normal = scipy.stats.norm.pdf(x)
    cdf_normal = scipy.stats.norm.cdf(x)

    # Calculate Student's t-distribution values for comparison
    pdf_t = scipy.stats.t.pdf(x, df=5)
    cdf_t = scipy.stats.t.cdf(x, df=5)

    # Plot the distributions on the same axes
    if show_pdf:
        ax1.plot(x, pdf_custom, label="Custom Mixture PDF", color="blue")
        ax1.plot(x, pdf_normal, label="Normal PDF", linestyle="--", color="green")
        ax1.plot(x, pdf_t, label="Student's t PDF", linestyle="--", color="red")
        ax1.set_title("PDF Comparison")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Density")

    if show_cdf:
        ax2.plot(x, cdf_custom, label="Custom Mixture CDF", color="blue")
        ax2.plot(x, cdf_normal, label="Normal CDF", linestyle="--", color="green")
        ax2.plot(x, cdf_t, label="Student's t CDF", linestyle="--", color="red")
        ax2.set_title("CDF Comparison")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Cumulative Probability")

    # Add legends, labels, and titles
    ax1.legend()
    ax2.legend()
    fig.tight_layout()

    # Return figure and axes for further customization
    return fig, axes


def main():
    """
    Main function demonstrating the use of the custom distribution.
    """
    # Generate sample data for testing
    np.random.seed(42)
    sample_size = 1000
    data = custom_random(sample_size, mu1=0, sigma1=1, mu2=3, sigma2=1, weight=0.7)

    # Create MixtureDistribution instance with example parameters
    mixture_dist = MixtureDistribution(mu1=0, sigma1=1, mu2=3, sigma2=1, weight=0.7)

    # Calculate PDF, CDF values for various inputs
    x_values = np.linspace(-5, 8, 400)
    pdf_values = mixture_dist.pdf(x_values)
    cdf_values = mixture_dist.cdf(x_values)

    # Generate random samples from the distribution
    random_samples = mixture_dist.random(size=sample_size)

    # Estimate distribution parameters from sample data
    estimated_params = estimate_custom_params(data)
    print("\nEstimated Parameters:", estimated_params)

    # Plot the custom distribution compared to standard distributions
    fig, axes = plot_distribution_comparison(estimated_params, x_values)
    plt.show()

    # Demonstrate fitting the distribution to data
    fitted_dist = mixture_dist.fit(data)
    print("\nFitted Distribution:", fitted_dist.__dict__)

    # Show use of Numba optimization for performance
    print("\nDemonstrating Numba optimization...")
    # The custom_pdf and custom_cdf functions are already decorated with @optimized_jit,
    # so they will be automatically optimized by Numba when called.


# Run the main function if the script is executed
if __name__ == "__main__":
    main()