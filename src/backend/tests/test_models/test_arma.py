"""
MFE Toolbox - ARMA Model Test Suite

This module contains comprehensive tests for the ARMA (AutoRegressive Moving Average)
model implementation in the MFE Toolbox. It includes tests for parameter estimation,
forecasting, diagnostic tools, and asynchronous operations with validation against
expected statistical properties.
"""

import pytest  # pytest 7.4.3
import numpy as np  # numpy 1.26.3
import pytest_asyncio  # pytest-asyncio 0.21.1
from hypothesis import given, strategies as st  # hypothesis 6.92.1
import scipy.stats as stats  # scipy 1.11.4
import statsmodels.api as sm  # statsmodels 0.14.1
import pandas as pd  # pandas 2.1.4
import asyncio  # Python 3.12

# Internal imports
from mfe.models.arma import ARMA, arma_order_select, compute_arma_residuals, arma_forecast  # src/backend/mfe/models/arma.py
from mfe.utils.numpy_helpers import ensure_array  # src/backend/mfe/utils/numpy_helpers.py
from mfe.utils.validation import validate_array  # src/backend/mfe/utils/validation.py
from . import parametrize_model_tests, create_model_test_data  # src/backend/tests/test_models/__init__.py

# Mark all tests in this module with 'models' and 'time_series' markers
pytestmark = [pytest.mark.models, pytest.mark.time_series]

# Define test configurations for ARMA models
ARMA_TEST_CONFIGS = [
    {'p': 1, 'q': 0, 'include_constant': True},
    {'p': 0, 'q': 1, 'include_constant': False},
    {'p': 1, 'q': 1, 'include_constant': True},
    {'p': 2, 'q': 1, 'include_constant': False}
]

def test_arma_initialization():
    """Test ARMA model initialization with various parameter configurations"""
    # Create ARMA models with different order configurations
    model_1 = ARMA(p=1, q=0, include_constant=True)
    model_2 = ARMA(p=0, q=1, include_constant=False)
    model_3 = ARMA(p=2, q=2, include_constant=True)

    # Verify initialization parameters are correctly set
    assert model_1.p == 1
    assert model_1.q == 0
    assert model_1.include_constant is True

    assert model_2.p == 0
    assert model_2.q == 1
    assert model_2.include_constant is False

    assert model_3.p == 2
    assert model_3.q == 2
    assert model_3.include_constant is True

    # Check edge cases like zero orders
    model_4 = ARMA(p=0, q=0, include_constant=False)
    assert model_4.p == 0
    assert model_4.q == 0
    assert model_4.include_constant is False

    # Verify error handling for invalid parameters
    with pytest.raises(ValueError):
        ARMA(p=-1, q=0)  # Negative AR order
    with pytest.raises(ValueError):
        ARMA(p=0, q=-1)  # Negative MA order

def test_arma_parameter_validation():
    """Test input validation for ARMA model parameters"""
    # Test with invalid AR order (negative)
    with pytest.raises(ValueError, match="AR and MA orders must be non-negative"):
        ARMA(p=-1, q=0)

    # Test with invalid MA order (negative)
    with pytest.raises(ValueError, match="AR and MA orders must be non-negative"):
        ARMA(p=0, q=-1)

    # Test with invalid optimizer string
    with pytest.warns(UserWarning, match="Optimizer 'invalid_optimizer' may not be supported"):
        ARMA(p=1, q=1, optimizer='invalid_optimizer')

def test_compute_arma_residuals():
    """Test the computation of ARMA model residuals"""
    # Create synthetic data with known properties
    np.random.seed(42)
    y = np.random.randn(100)
    ar_params = np.array([0.5])
    ma_params = np.array([0.3])
    constant = 0.1

    # Compute residuals with known parameters
    residuals = compute_arma_residuals(y, ar_params, ma_params, constant)

    # Compare with expected residuals (basic check)
    assert residuals.shape == y.shape
    assert not np.isnan(residuals).any()

    # Test different AR and MA parameter combinations
    residuals_no_ar = compute_arma_residuals(y, np.array([]), ma_params, constant)
    residuals_no_ma = compute_arma_residuals(y, ar_params, np.array([]), constant)
    residuals_no_constant = compute_arma_residuals(y, ar_params, ma_params, 0.0)

    assert residuals_no_ar.shape == y.shape
    assert residuals_no_ma.shape == y.shape
    assert residuals_no_constant.shape == y.shape

@parametrize_model_tests(ARMA_TEST_CONFIGS, include_numba_tests=True)
def test_arma_estimation(model_config):
    """Test ARMA model parameter estimation"""
    # Create test time series data
    np.random.seed(42)
    y = create_model_test_data(model_type='arma', sample_size=200, properties=model_config)
    
    # Initialize ARMA model with test configuration
    model = ARMA(**model_config)

    # Estimate model parameters
    model.estimate(y)

    # Verify parameters are within expected bounds
    assert model.loglikelihood is not None
    assert model.sigma2 is not None
    assert model.residuals is not None

    # Verify model diagnostics are computed correctly
    assert model.summary() is not None

    # Check residuals for white noise properties
    jb_stat, jb_pval = model.diagnostic_tests()['jarque_bera']
    assert jb_pval > 0.05  # Should not reject normality

def test_arma_with_exogenous():
    """Test ARMA model with exogenous variables"""
    # Create test time series with exogenous variables
    np.random.seed(42)
    y = np.random.randn(100)
    exog = np.random.randn(100, 2)  # Two exogenous variables

    # Estimate ARMA model with exogenous variables
    model = ARMA(p=1, q=1, include_constant=True)
    model.estimate(y, exog=exog)

    # Verify exogenous parameters are estimated correctly
    assert model.exog_params is not None
    assert len(model.exog_params) == 2

    # Test forecasting with exogenous variables
    exog_forecast = np.random.randn(5, 2)
    forecasts = model.forecast(y, steps=5, exog_forecast=exog_forecast)
    assert len(forecasts) == 5

@parametrize_model_tests(ARMA_TEST_CONFIGS, include_numba_tests=True)
def test_arma_forecast(model_config):
    """Test ARMA model forecasting"""
    # Create and estimate ARMA model
    np.random.seed(42)
    y = create_model_test_data(model_type='arma', sample_size=200, properties=model_config)
    model = ARMA(**model_config)
    model.estimate(y)

    # Generate forecasts for multiple steps ahead
    forecasts = model.forecast(y, steps=10)

    # Verify forecast shapes and values
    assert len(forecasts) == 10
    assert not np.isnan(forecasts).any()

    # Test forecast variance calculation
    forecast_variance = model.forecast_variance(steps=10)
    assert len(forecast_variance) == 10
    assert not np.isnan(forecast_variance).any()

    # Compare forecasts to analytical expectations (basic check)
    if model.q == 0:  # Pure AR model
        assert np.all(np.abs(forecasts) < 5)  # Forecasts should not explode

def test_arma_order_select():
    """Test automatic ARMA order selection"""
    # Generate data from known ARMA process
    np.random.seed(42)
    ar_params = np.array([0.6, 0.2])
    ma_params = np.array([0.4])
    y = create_model_test_data(model_type='arma', sample_size=200, properties={'ar_coefs': ar_params, 'ma_coefs': ma_params})

    # Use arma_order_select to determine optimal orders
    best_p, best_q = arma_order_select(y, max_ar=3, max_ma=3, ic='aic')

    # Verify selected orders match the known true orders
    assert best_p == 2
    assert best_q == 1

    # Test with different information criteria
    best_bic_p, best_bic_q = arma_order_select(y, max_ar=3, max_ma=3, ic='bic')
    assert best_bic_p == 0 or best_bic_p == 2
    assert best_bic_q == 0 or best_bic_q == 1

def test_arma_simulate():
    """Test ARMA model simulation"""
    # Create ARMA model with fixed parameters
    model = ARMA(p=1, q=1, include_constant=True)
    model.ar_params = np.array([0.5])
    model.ma_params = np.array([0.3])
    model.constant = 0.1
    model.sigma2 = 1.0

    # Simulate time series data
    y_sim = model.simulate(nsimulations=1000, burn=100)

    # Verify statistical properties of the simulated data
    assert len(y_sim) == 1000
    assert not np.isnan(y_sim).any()

    # Test with various parameter configurations
    model_no_constant = ARMA(p=1, q=1, include_constant=False)
    model_no_constant.ar_params = np.array([0.5])
    model_no_constant.ma_params = np.array([0.3])
    model_no_constant.sigma2 = 1.0
    y_sim_no_constant = model_no_constant.simulate(nsimulations=1000, burn=100)
    assert len(y_sim_no_constant) == 1000

def test_arma_diagnostic_tests():
    """Test diagnostic tests for ARMA models"""
    # Create test time series data
    np.random.seed(42)
    y = np.random.randn(200)

    # Estimate ARMA model with test data
    model = ARMA(p=1, q=1, include_constant=True)
    model.estimate(y)

    # Perform various diagnostic tests
    diagnostic_results = model.diagnostic_tests()

    # Verify test results are correctly computed
    assert 'ljung_box' in diagnostic_results
    assert 'jarque_bera' in diagnostic_results

    # Check test implementations against statsmodels
    sm_lb_results = sm.stats.diagnostic.acorr_ljungbox(model.residuals, lags=[10], return_df=True)
    assert diagnostic_results['ljung_box']['statistic'] == pytest.approx(sm_lb_results['lb_stat'].values[0], rel=0.01)

@pytest.mark.asyncio
async def test_async_estimation():
    """Test asynchronous ARMA model parameter estimation"""
    # Create test time series data
    np.random.seed(42)
    y = np.random.randn(200)

    # Initialize ARMA model
    model = ARMA(p=1, q=1, include_constant=True)

    # Estimate parameters asynchronously using estimate_async
    await model.estimate_async(y)

    # Verify results match synchronous estimation
    assert model.loglikelihood is not None
    assert model.sigma2 is not None
    assert model.residuals is not None

    # Test with progress reporting
    async def progress_reporter(progress):
        assert 0 <= progress <= 100

    await model.estimate_async(y)

@pytest.mark.asyncio
async def test_async_forecast():
    """Test asynchronous ARMA model forecasting"""
    # Create test time series data
    np.random.seed(42)
    y = np.random.randn(200)

    # Estimate ARMA model with test data
    model = ARMA(p=1, q=1, include_constant=True)
    model.estimate(y)

    # Generate forecasts asynchronously
    forecasts = await model.forecast_async(y, steps=10)

    # Verify results match synchronous forecasting
    assert len(forecasts) == 10
    assert not np.isnan(forecasts).any()

    # Test multiple forecast horizons
    forecasts_20 = await model.forecast_async(y, steps=20)
    assert len(forecasts_20) == 20

def test_model_persistence():
    """Test saving and loading ARMA models"""
    # Create and estimate ARMA model
    np.random.seed(42)
    y = np.random.randn(200)
    model = ARMA(p=1, q=1, include_constant=True)
    model.estimate(y)

    # Convert model to dictionary using to_dict
    model_dict = model.to_dict()

    # Create new model from dictionary using from_dict
    new_model = ARMA.from_dict(model_dict)

    # Verify models are equivalent
    assert new_model.p == model.p
    assert new_model.q == model.q
    assert new_model.include_constant == model.include_constant
    assert np.allclose(new_model.ar_params, model.ar_params)
    assert np.allclose(new_model.ma_params, model.ma_params)
    assert new_model.constant == model.constant
    assert new_model.sigma2 == model.sigma2
    assert new_model.loglikelihood == model.loglikelihood

    # Test with different model configurations
    model_no_constant = ARMA(p=2, q=0, include_constant=False)
    model_no_constant.estimate(y)
    model_dict_no_constant = model_no_constant.to_dict()
    new_model_no_constant = ARMA.from_dict(model_dict_no_constant)
    assert new_model_no_constant.include_constant is False

@given(st.floats(min_value=-0.9, max_value=0.9))
def test_property_stationarity(ar_coef):
    """Property-based test for ARMA stationarity"""
    # Create ARMA models with parameters from hypothesis
    model = ARMA(p=1, q=0, include_constant=True)
    model.ar_params = np.array([ar_coef])

    # Verify stationarity conditions are correctly evaluated
    if abs(ar_coef) < 1:
        assert model.is_stationary() is True
    else:
        assert model.is_stationary() is False

@given(st.floats(min_value=-0.9, max_value=0.9))
def test_property_invertibility(ma_coef):
    """Property-based test for ARMA invertibility"""
    # Create ARMA models with parameters from hypothesis
    model = ARMA(p=0, q=1, include_constant=True)
    model.ma_params = np.array([ma_coef])

    # Verify invertibility conditions are correctly evaluated
    if abs(ma_coef) < 1:
        assert model.is_invertible() is True
    else:
        assert model.is_invertible() is False