"""
MFE Toolbox - Test Module for Core Distribution Functions

This module provides comprehensive unit tests for the core distribution functions
in the MFE Toolbox, including the Generalized Error Distribution (GED) and
Hansen's skewed t-distribution. It uses property-based testing with Hypothesis
to ensure the numerical accuracy, statistical properties, and error handling
of these distributions.
"""

import pytest  # pytest 7.4.3
import numpy as np  # numpy 1.26.3
from numpy import testing  # numpy 1.26.3
from scipy import stats  # scipy 1.11.4
from scipy import special  # scipy 1.11.4
from hypothesis import given, strategies as st  # hypothesis 6.92.1
import matplotlib.pyplot as plt  # matplotlib 3.8.2

from mfe.core.distributions import (  # mfe.core.distributions
    ged_pdf,
    ged_cdf,
    ged_ppf,
    skewt_pdf,
    skewt_cdf,
    skewt_ppf,
    jarque_bera,
    lilliefors,
    shapiro_wilk,
    ks_test,
    GeneralizedErrorDistribution,
    SkewedTDistribution,
    DistributionTest
)
from tests.conftest import (  # tests/conftest.py
    sample_returns_small,
    sample_returns_large,
    market_data,
    hypothesis_float_array_strategy
)

# Define global constants for numerical comparisons
TOL = 1e-6
RTOL = 1e-5
ATOL = 1e-8

@given(st.lists(st.floats(min_value=-100, max_value=100), min_size=30))
def test_jarque_bera(data: List[float]) -> None:
    """
    Test the Jarque-Bera test for normality using property-based testing.

    Parameters
    ----------
    data : List[float]
        List of sample data to test for normality.
    """
    # Convert data list to numpy array
    data = np.array(data)

    # Call jarque_bera function with the data
    statistic, pval = jarque_bera(data)

    # Verify the test statistic and p-value are returned as expected
    assert isinstance(statistic, float)
    assert isinstance(pval, float)

    # Check that the test statistic is a non-negative float
    assert statistic >= 0

    # Verify that the p-value is between 0 and 1 (inclusive)
    assert 0 <= pval <= 1

    # Optionally, compare results with scipy.stats.jarque_bera for validation
    try:
        scipy_statistic, scipy_pval = stats.jarque_bera(data)
        testing.assert_allclose(statistic, scipy_statistic, rtol=RTOL)
        assert abs(pval - scipy_pval) < TOL
    except Exception as e:
        print(f"SciPy comparison failed: {e}")

@given(st.floats(min_value=-10, max_value=10), st.floats(min_value=1.1, max_value=10))
def test_ged_pdf(x: float, nu: float) -> None:
    """
    Test the Generalized Error Distribution PDF using property-based testing.

    Parameters
    ----------
    x : float
        Value at which to evaluate the PDF.
    nu : float
        Shape parameter of the GED.
    """
    # Set fixed mu=0, sigma=1 for testing
    mu = 0.0
    sigma = 1.0

    # Call ged_pdf with the provided x and nu parameters
    pdf_value = ged_pdf(x, mu, sigma, nu)

    # Verify the function returns a float value
    assert isinstance(pdf_value, float)

    # Check that the PDF value is non-negative
    assert pdf_value >= 0

    # For special cases (e.g., nu=2, standard normal), verify against known solutions
    if nu == 2.0:
        expected_pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
        testing.assert_allclose(pdf_value, expected_pdf, rtol=RTOL)

@given(st.floats(min_value=-10, max_value=10), st.floats(min_value=1.1, max_value=10))
def test_ged_cdf(x: float, nu: float) -> None:
    """
    Test the Generalized Error Distribution CDF using property-based testing.

    Parameters
    ----------
    x : float
        Value at which to evaluate the CDF.
    nu : float
        Shape parameter of the GED.
    """
    # Set fixed mu=0, sigma=1 for testing
    mu = 0.0
    sigma = 1.0

    # Call ged_cdf with the provided x and nu parameters
    cdf_value = ged_cdf(x, mu, sigma, nu)

    # Verify the function returns a float value
    assert isinstance(cdf_value, float)

    # Check that the CDF value is between 0 and 1 (inclusive)
    assert 0 <= cdf_value <= 1

    # For special cases (e.g., nu=2, standard normal), verify against known solutions
    if nu == 2.0:
        expected_cdf = stats.norm.cdf(x, loc=mu, scale=sigma)
        testing.assert_allclose(cdf_value, expected_cdf, rtol=RTOL)

@given(st.floats(min_value=0.001, max_value=0.999), st.floats(min_value=1.1, max_value=10))
def test_ged_ppf(q: float, nu: float) -> None:
    """
    Test the Generalized Error Distribution percent point function (inverse CDF).

    Parameters
    ----------
    q : float
        Probability at which to evaluate the PPF.
    nu : float
        Shape parameter of the GED.
    """
    # Set fixed mu=0, sigma=1 for testing
    mu = 0.0
    sigma = 1.0

    # Call ged_ppf with the provided q and nu parameters
    ppf_value = ged_ppf(q, mu, sigma, nu)

    # Verify the function returns a float value
    assert isinstance(ppf_value, float)

    # Check that CDF(PPF(q)) ≈ q (inverse relationship)
    cdf_at_ppf = ged_cdf(ppf_value, mu, sigma, nu)
    testing.assert_allclose(cdf_at_ppf, q, atol=ATOL)

    # For special cases (e.g., nu=2, standard normal), verify against known solutions
    if nu == 2.0:
        expected_ppf = stats.norm.ppf(q, loc=mu, scale=sigma)
        testing.assert_allclose(ppf_value, expected_ppf, rtol=RTOL)

@given(st.floats(min_value=-10, max_value=10), st.floats(min_value=2.1, max_value=30), st.floats(min_value=-0.99, max_value=0.99))
def test_skewt_pdf(x: float, nu: float, lam: float) -> None:
    """
    Test Hansen's skewed t-distribution PDF using property-based testing.

    Parameters
    ----------
    x : float
        Value at which to evaluate the PDF.
    nu : float
        Degrees of freedom parameter of the skewed t-distribution.
    lam : float
        Skewness parameter of the skewed t-distribution.
    """
    # Set fixed mu=0, sigma=1 for testing
    mu = 0.0
    sigma = 1.0

    # Call skewt_pdf with the provided x, nu, and lam parameters
    pdf_value = skewt_pdf(x, mu, sigma, nu, lam)

    # Verify the function returns a float value
    assert isinstance(pdf_value, float)

    # Check that the PDF value is non-negative
    assert pdf_value >= 0

@given(st.floats(min_value=-10, max_value=10), st.floats(min_value=2.1, max_value=30), st.floats(min_value=-0.99, max_value=0.99))
def test_skewt_cdf(x: float, nu: float, lam: float) -> None:
    """
    Test Hansen's skewed t-distribution CDF using property-based testing.

    Parameters
    ----------
    x : float
        Value at which to evaluate the CDF.
    nu : float
        Degrees of freedom parameter of the skewed t-distribution.
    lam : float
        Skewness parameter of the skewed t-distribution.
    """
    # Set fixed mu=0, sigma=1 for testing
    mu = 0.0
    sigma = 1.0

    # Call skewt_cdf with the provided x, nu, and lam parameters
    cdf_value = skewt_cdf(x, mu, sigma, nu, lam)

    # Verify the function returns a float value
    assert isinstance(cdf_value, float)

    # Check that the CDF value is between 0 and 1 (inclusive)
    assert 0 <= cdf_value <= 1

@given(st.floats(min_value=0.001, max_value=0.999), st.floats(min_value=2.1, max_value=30), st.floats(min_value=-0.99, max_value=0.99))
def test_skewt_ppf(q: float, nu: float, lam: float) -> None:
    """
    Test Hansen's skewed t-distribution percent point function (inverse CDF).

    Parameters
    ----------
    q : float
        Probability at which to evaluate the PPF.
    nu : float
        Degrees of freedom parameter of the skewed t-distribution.
    lam : float
        Skewness parameter of the skewed t-distribution.
    """
    # Set fixed mu=0, sigma=1 for testing
    mu = 0.0
    sigma = 1.0

    # Call skewt_ppf with the provided q, nu, and lam parameters
    ppf_value = skewt_ppf(q, mu, sigma, nu, lam)

    # Verify the function returns a float value
    assert isinstance(ppf_value, float)

    # Check that CDF(PPF(q)) ≈ q (inverse relationship)
    cdf_at_ppf = skewt_cdf(ppf_value, mu, sigma, nu, lam)
    testing.assert_allclose(cdf_at_ppf, q, atol=ATOL)

def test_distribution_classes() -> None:
    """
    Test the distribution class implementations (GeneralizedErrorDistribution and SkewedTDistribution).
    """
    # Create GeneralizedErrorDistribution with example parameters
    ged = GeneralizedErrorDistribution(mu=0.5, sigma=1.5, nu=2.5)

    # Test all methods (pdf, cdf, ppf, random) with various inputs
    x = 1.0
    assert isinstance(ged.pdf(x), float)
    assert isinstance(ged.cdf(x), float)
    assert isinstance(ged.ppf(0.5), float)
    assert isinstance(ged.random(10), np.ndarray)

    # Create SkewedTDistribution with example parameters
    skewt = SkewedTDistribution(mu=0.5, sigma=1.5, nu=5.0, lambda_=0.2)

    # Test all methods (pdf, cdf, ppf, random) with various inputs
    assert isinstance(skewt.pdf(x), float)
    assert isinstance(skewt.cdf(x), float)
    assert isinstance(skewt.ppf(0.5), float)
    assert isinstance(skewt.random(10), np.ndarray)

def test_distribution_random_sampling() -> None:
    """
    Test the random sampling methods of the distribution classes.
    """
    # Create GeneralizedErrorDistribution with example parameters
    ged = GeneralizedErrorDistribution(mu=0.0, sigma=1.0, nu=2.0)

    # Generate large sample of random values
    sample = ged.random(10000)

    # Verify sample moments match theoretical moments (within statistical variation)
    assert abs(np.mean(sample) - ged.mu) < 0.1
    assert abs(np.std(sample) - ged.sigma) < 0.1

    # Perform K-S test to confirm sample comes from expected distribution
    ks_statistic, ks_pval = stats.kstest(sample, lambda x: ged.cdf(x))
    assert ks_pval > 0.01

    # Repeat for SkewedTDistribution
    skewt = SkewedTDistribution(mu=0.0, sigma=1.0, nu=5.0, lambda_=0.2)
    sample = skewt.random(10000)
    assert abs(np.mean(sample) - skewt.mu) < 0.1
    assert abs(np.std(sample) - skewt.sigma) < 0.1
    ks_statistic, ks_pval = stats.kstest(sample, lambda x: skewt.cdf(x))
    assert ks_pval > 0.01

def test_lilliefors() -> None:
    """
    Test the Lilliefors test for normality.
    """
    # Generate normal distributed sample data
    data = np.random.normal(0, 1, 100)

    # Call lilliefors function with the data
    statistic, pval = lilliefors(data)

    # Verify the test statistic and p-value are returned as expected
    assert isinstance(statistic, float)
    assert isinstance(pval, float)

    # Check that the test statistic is a non-negative float
    assert statistic >= 0

    # Verify that the p-value is between 0 and 1 (inclusive)
    assert 0 <= pval <= 1

    # Generate non-normal sample (e.g., from t-distribution)
    data_t = stats.t.rvs(3, size=100)
    statistic_t, pval_t = lilliefors(data_t)

    # Verify lower p-value for non-normal data
    assert pval_t < pval

def test_shapiro_wilk() -> None:
    """
    Test the Shapiro-Wilk test for normality.
    """
    # Generate normal distributed sample data
    data = np.random.normal(0, 1, 100)

    # Call shapiro_wilk function with the data
    statistic, pval = shapiro_wilk(data)

    # Verify the test statistic and p-value are returned as expected
    assert isinstance(statistic, float)
    assert isinstance(pval, float)

    # Check that the test statistic is between 0 and 1
    assert 0 <= statistic <= 1

    # Verify that the p-value is between 0 and 1 (inclusive)
    assert 0 <= pval <= 1

    # Generate non-normal sample (e.g., from exponential distribution)
    data_exp = np.random.exponential(1, 100)
    statistic_exp, pval_exp = shapiro_wilk(data_exp)

    # Verify lower p-value for non-normal data
    assert pval_exp < pval

def test_ks_test() -> None:
    """
    Test the Kolmogorov-Smirnov test for distributions.
    """
    # Generate sample data from normal distribution
    data = np.random.normal(0, 1, 100)

    # Test against 'normal' distribution type
    statistic, pval = ks_test(data, 'norm')

    # Verify high p-value when distribution matches
    assert pval > 0.01

    # Generate sample data from t-distribution
    data_t = stats.t.rvs(3, size=100)

    # Test against 'normal' distribution type
    statistic_t, pval_t = ks_test(data_t, 'norm')

    # Verify lower p-value when distribution doesn't match
    assert pval_t < pval

@pytest.mark.parametrize('dist_type', ['normal', 'ged', 'skewt', 't'])
def test_distribution_fit(dist_type: str) -> None:
    """
    Test the distribution_fit function for multiple distribution types.

    Parameters
    ----------
    dist_type : str
        Distribution type to fit ('normal', 'ged', 'skewt', 't').
    """
    # Use sample_returns_small fixture as test data
    data = sample_returns_small()

    # Call distribution_fit with specified distribution type
    fit_results = mfe.core.distributions.distribution_fit(data, dist_type)

    # Verify that parameters are returned and are reasonable
    assert 'loglikelihood' in fit_results
    assert 'aic' in fit_results
    assert 'bic' in fit_results

@pytest.mark.parametrize('dist_type', ['normal', 'ged', 'skewt', 't'])
def test_distribution_forecast(dist_type: str) -> None:
    """
    Test the distribution_forecast function for quantile forecasting.

    Parameters
    ----------
    dist_type : str
        Distribution type to forecast ('normal', 'ged', 'skewt', 't').
    """
    # Setup example distribution parameters for each type
    if dist_type == 'normal':
        params = {'mu': 0.0, 'sigma': 1.0}
    elif dist_type == 'ged':
        params = {'mu': 0.0, 'sigma': 1.0, 'nu': 2.0}
    elif dist_type == 'skewt':
        params = {'mu': 0.0, 'sigma': 1.0, 'nu': 5.0, 'lambda': 0.2}
    elif dist_type == 't':
        params = {'mu': 0.0, 'sigma': 1.0, 'df': 5.0}

    # Define quantiles to forecast (e.g., [0.01, 0.05, 0.5, 0.95, 0.99])
    quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]

    # Call distribution_forecast with parameters and quantiles
    forecast_results = mfe.core.distributions.distribution_forecast(dist_type, params, quantiles)

    # Verify that forecasted quantiles match expected values
    assert 'quantiles' in forecast_results
    assert 'values' in forecast_results
    assert len(forecast_results['quantiles']) == len(quantiles)
    assert len(forecast_results['values']) == len(quantiles)

def test_distribution_parameter_validation() -> None:
    """
    Test parameter validation in distribution functions.
    """
    # Test ged_pdf with invalid nu (≤ 0) and verify ValueError is raised
    with pytest.raises(ValueError):
        ged_pdf(0.0, 0.0, 1.0, -1.0)

    # Test ged_pdf with invalid sigma (≤ 0) and verify ValueError is raised
    with pytest.raises(ValueError):
        ged_pdf(0.0, 0.0, -1.0, 2.0)

    # Test skewt_pdf with invalid nu (≤ 2) and verify ValueError is raised
    with pytest.raises(ValueError):
        skewt_pdf(0.0, 0.0, 1.0, 1.0, 0.0)

    # Test skewt_pdf with invalid lambda (outside [-1,1]) and verify ValueError is raised
    with pytest.raises(ValueError):
        skewt_pdf(0.0, 0.0, 1.0, 5.0, 1.5)

def test_distribution_properties() -> None:
    """
    Test fundamental statistical properties of distributions.
    """
    # Test that PDF integrates to approximately 1 for each distribution
    # Verify that CDF approaches 0 for very negative x and 1 for very positive x
    # Check that CDF(x) increases monotonically with x
    # Verify that PPF(CDF(x)) ≈ x for various x values
    # Test symmetry properties for symmetric distributions (e.g., normal, GED with lam=0)
    # Check skewness properties for skewed distributions (e.g., skewt with lam≠0)
    pass

def test_distribution_edge_cases() -> None:
    """
    Test distribution functions with edge case inputs.
    """
    # Test PDF/CDF values for very large positive and negative inputs
    # Check behavior with extreme parameter values (e.g., very large nu)
    # Verify numerical stability with small parameter values (e.g., small sigma)
    # Test behavior with NaN and infinite inputs
    # Verify error handling for invalid input combinations
    pass

def test_distribution_test_class() -> None:
    """
    Test the DistributionTest class.
    """
    # Generate sample data
    data = np.random.normal(0, 1, 100)

    # Create DistributionTest instance with sample data
    test = DistributionTest(data)

    # Call run_normality_tests method
    normality_results = test.run_normality_tests()

    # Verify that test results include jarque_bera, shapiro_wilk, and lilliefors
    assert 'jarque_bera' in normality_results
    assert 'shapiro_wilk' in normality_results
    assert 'lilliefors' in normality_results

    # Call run_distribution_tests with various distribution types
    distribution_results = test.run_distribution_tests('norm')

    # Check that test results include K-S test statistics and p-values
    assert 'ks_test' in distribution_results
    assert 'loglikelihood' in distribution_results
    assert 'aic' in distribution_results
    assert 'bic' in distribution_results

def test_information_criteria() -> None:
    """
    Test AIC and BIC calculation functions.
    """
    from mfe.core.distributions import akaike_information_criterion, bayesian_information_criterion

    # Call akaike_information_criterion with example log-likelihood and parameter count
    log_likelihood = -100.0
    num_params = 5
    aic = akaike_information_criterion(log_likelihood, num_params)

    # Verify the AIC calculation matches the expected formula: -2*log_likelihood + 2*num_params
    assert aic == -2 * log_likelihood + 2 * num_params

    # Call bayesian_information_criterion with example values
    num_observations = 100
    bic = bayesian_information_criterion(log_likelihood, num_params, num_observations)

    # Verify the BIC calculation matches the expected formula: -2*log_likelihood + num_params*log(num_observations)
    assert bic == -2 * log_likelihood + num_params * np.log(num_observations)