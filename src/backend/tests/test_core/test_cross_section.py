"""
Test suite for the cross-section module which provides regression and principal component analysis
functionality for the MFE Toolbox.
"""

import pytest
import numpy as np
import scipy.stats
import statsmodels.api as sm
from hypothesis import given, strategies as st

from mfe.core import cross_section


def generate_regression_data(n_samples=100, n_features=3, noise=0.1):
    """Helper function to generate regression test data with known coefficients."""
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(n_samples, n_features)
    true_coefs = np.random.randn(n_features)
    y = X @ true_coefs + noise * np.random.randn(n_samples)
    return X, y, true_coefs


def generate_pca_data(n_samples=100, n_features=5, n_components=2):
    """Helper function to generate data with known principal components."""
    np.random.seed(42)  # For reproducibility
    
    # Generate random component directions
    components = np.random.randn(n_components, n_features)
    
    # Normalize components
    for i in range(n_components):
        components[i] = components[i] / np.linalg.norm(components[i])
    
    # Generate random weights for each sample
    weights = np.random.randn(n_samples, n_components)
    
    # Create data by combining components with weights
    X = weights @ components
    
    # Add some noise
    X += 0.1 * np.random.randn(n_samples, n_features)
    
    return X, components


def test_linear_regression_basic():
    """Test basic functionality of linear regression with simple dataset."""
    # Generate test data
    X, y, true_coefs = generate_regression_data(n_samples=100, n_features=3, noise=0.1)
    
    # Add constant to X for intercept
    X_with_const = sm.add_constant(X)
    
    # Run regression
    result = cross_section.cross_sectional_regression(y, X_with_const)
    
    # Verify result contains expected keys
    assert 'params' in result
    assert 'std_errors' in result
    assert 't_stats' in result
    assert 'p_values' in result
    assert 'r_squared' in result
    assert 'adj_r_squared' in result
    
    # Verify result dimensions
    assert len(result['params']) == X_with_const.shape[1]
    
    # Verify R-squared is between 0 and 1
    assert 0 <= result['r_squared'] <= 1
    
    # Verify model predicts reasonably well (R-squared > 0.8 for this low-noise data)
    assert result['r_squared'] > 0.8


def test_linear_regression_edge_cases():
    """Test linear regression with edge cases like single predictor or perfect multicollinearity."""
    # Test with single predictor
    X, y, _ = generate_regression_data(n_samples=50, n_features=1, noise=0.1)
    X_with_const = sm.add_constant(X)
    result = cross_section.cross_sectional_regression(y, X_with_const)
    assert 'params' in result
    assert len(result['params']) == 2  # Intercept + 1 feature
    
    # Test with perfect multicollinearity
    X, y, _ = generate_regression_data(n_samples=50, n_features=2, noise=0.1)
    X_with_const = sm.add_constant(X)
    # Make last column a copy of the second column to create perfect multicollinearity
    X_with_const = np.column_stack([X_with_const, X_with_const[:, 1]])
    
    # This should not raise an error
    result = cross_section.cross_sectional_regression(y, X_with_const)
    
    # Test with constant predictor
    X = np.ones((50, 1))  # Column of ones
    X_with_const = sm.add_constant(X)
    y = np.random.randn(50)
    
    # This should not raise an error
    result = cross_section.cross_sectional_regression(y, X_with_const)


def test_linear_regression_with_nan():
    """Test linear regression handling of missing values."""
    # Generate data with some NaN values
    X, y, _ = generate_regression_data(n_samples=100, n_features=3, noise=0.1)
    X_with_const = sm.add_constant(X)
    
    # Set some values to NaN
    X_with_nan = X_with_const.copy()
    X_with_nan[5, 1] = np.nan
    y_with_nan = y.copy()
    y_with_nan[10] = np.nan
    
    # This should raise an Exception
    with pytest.raises(Exception):
        cross_section.cross_sectional_regression(y_with_nan, X_with_const)
    
    with pytest.raises(Exception):
        cross_section.cross_sectional_regression(y, X_with_nan)


def test_robust_regression():
    """Test robust regression functionality with outliers in the dataset."""
    # Generate clean data
    X, y, _ = generate_regression_data(n_samples=100, n_features=3, noise=0.1)
    X_with_const = sm.add_constant(X)
    
    # Add outliers
    y_with_outliers = y.copy()
    outlier_indices = [5, 10, 15]
    y_with_outliers[outlier_indices] += 10  # Add large errors
    
    # Run standard OLS regression
    result_ols = cross_section.cross_sectional_regression(y_with_outliers, X_with_const, robust=False)
    
    # Run robust regression
    result_robust = cross_section.cross_sectional_regression(y_with_outliers, X_with_const, robust=True)
    
    # Verify both methods return results
    assert 'params' in result_ols
    assert 'params' in result_robust
    
    # Verify standard errors are different between the methods
    assert not np.allclose(result_ols['std_errors'], result_robust['std_errors'])
    
    # Run with different robust covariance types
    result_hc0 = cross_section.cross_sectional_regression(y_with_outliers, X_with_const, 
                                                        robust=True, cov_type='HC0')
    result_hc3 = cross_section.cross_sectional_regression(y_with_outliers, X_with_const, 
                                                        robust=True, cov_type='HC3')
    
    # Verify results are different between covariance types
    assert not np.allclose(result_hc0['std_errors'], result_hc3['std_errors'])
    
    # Test with CrossSectionalRegression class
    model_ols = cross_section.CrossSectionalRegression(robust=False)
    model_ols.fit(y_with_outliers, X)
    
    model_robust = cross_section.CrossSectionalRegression(robust=True)
    model_robust.fit(y_with_outliers, X)
    
    # Verify standard errors are different
    assert not np.allclose(model_ols.results_['std_errors'], model_robust.results_['std_errors'])


def test_pca_basic():
    """Test basic functionality of principal component analysis."""
    # Generate test data
    X, true_components = generate_pca_data(n_samples=100, n_features=5, n_components=2)
    
    # Run PCA
    result = cross_section.principal_component_analysis(X, n_components=2)
    
    # Verify result contains expected keys
    assert 'eigenvalues' in result
    assert 'eigenvectors' in result
    assert 'explained_variance' in result
    assert 'explained_variance_ratio' in result
    assert 'cumulative_variance_ratio' in result
    assert 'projected_data' in result
    
    # Verify dimensions
    assert len(result['eigenvalues']) == 2
    assert result['eigenvectors'].shape == (5, 2)
    assert result['projected_data'].shape == (100, 2)
    
    # Verify explained variance ratio sums to 1 (or close to it)
    assert np.isclose(np.sum(result['explained_variance_ratio']), 1.0, atol=1e-6)
    
    # Verify last value of cumulative variance ratio is 1 (or close to it)
    assert np.isclose(result['cumulative_variance_ratio'][-1], 1.0, atol=1e-6)
    
    # Test with different numbers of components
    result_3 = cross_section.principal_component_analysis(X, n_components=3)
    assert len(result_3['eigenvalues']) == 3
    assert result_3['eigenvectors'].shape == (5, 3)
    assert result_3['projected_data'].shape == (100, 3)
    
    # Test with default n_components (should use min(n_samples, n_features))
    result_default = cross_section.principal_component_analysis(X)
    assert len(result_default['eigenvalues']) == 5  # min(100, 5) = 5


def test_pca_scaling():
    """Test PCA with different scaling options."""
    # Generate data with variables of different scales
    np.random.seed(42)
    X = np.random.randn(100, 3)
    # Make the first column have much larger values
    X[:, 0] *= 100
    
    # Run PCA with and without standardization
    result_std = cross_section.principal_component_analysis(X, standardize=True)
    result_no_std = cross_section.principal_component_analysis(X, standardize=False)
    
    # Verify both return results
    assert 'eigenvectors' in result_std
    assert 'eigenvectors' in result_no_std
    
    # The results should be different when scaling is used vs. not used
    assert not np.allclose(result_std['eigenvectors'], result_no_std['eigenvectors'])
    
    # Without standardization, the first component should align more with the first variable
    # because it has much larger variance
    first_loading_no_std = np.abs(result_no_std['eigenvectors'][0, 0])
    assert first_loading_no_std > 0.9  # The first variable should dominate
    
    # With standardization, the loadings should be more balanced
    first_loading_std = np.abs(result_std['eigenvectors'][0, 0])
    assert first_loading_std < 0.9  # The first variable should not dominate as much
    
    # Test with PrincipalComponentAnalysis class
    pca_std = cross_section.PrincipalComponentAnalysis(standardize=True)
    pca_std.fit(X)
    
    pca_no_std = cross_section.PrincipalComponentAnalysis(standardize=False)
    pca_no_std.fit(X)
    
    # The components should be different
    assert not np.allclose(pca_std.components_, pca_no_std.components_)


def test_cross_validation_regression():
    """Test cross-validation functionality for regression models."""
    # Generate test data
    X, y, _ = generate_regression_data(n_samples=100, n_features=3, noise=0.1)
    
    # Create a simple cross-validation function
    def cross_validate(X, y, n_folds=5):
        # Simple k-fold cross-validation
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        fold_size = len(indices) // n_folds
        
        results = []
        
        for i in range(n_folds):
            # Create test and train indices
            test_idx = indices[i * fold_size:(i + 1) * fold_size]
            train_idx = np.array([idx for idx in indices if idx not in test_idx])
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Add constant for intercept
            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test)
            
            # Fit model
            model = cross_section.CrossSectionalRegression()
            model.fit(y_train, X_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = model.r_squared_
            mse = np.mean((y_pred - y_test) ** 2)
            
            results.append({'fold': i, 'r2': r2, 'mse': mse})
        
        return results
    
    # Perform cross-validation
    cv_results = cross_validate(X, y, n_folds=5)
    
    # Check that we have results for all folds
    assert len(cv_results) == 5
    
    # Check that metrics are reasonable
    r2_values = [result['r2'] for result in cv_results]
    mse_values = [result['mse'] for result in cv_results]
    
    # For this synthetic data, R² should generally be good
    assert np.mean(r2_values) > 0.7
    
    # MSE should be positive but relatively small
    assert all(mse > 0 for mse in mse_values)
    assert np.mean(mse_values) < 1.0  # Reasonable threshold for this data


def test_input_validation():
    """Test input validation in cross-sectional functions."""
    # Test with invalid input types
    with pytest.raises(Exception):
        cross_section.cross_sectional_regression("not_an_array", np.random.randn(10, 2))
    
    with pytest.raises(Exception):
        cross_section.cross_sectional_regression(np.random.randn(10), "not_an_array")
    
    with pytest.raises(Exception):
        cross_section.principal_component_analysis("not_an_array")
    
    # Test with mismatched dimensions
    with pytest.raises(Exception):
        cross_section.cross_sectional_regression(np.random.randn(10), np.random.randn(15, 2))
    
    # Test with invalid parameters
    with pytest.raises(Exception):
        cross_section.principal_component_analysis(np.random.randn(10, 3), n_components=-1)
    
    with pytest.raises(Exception):
        cross_section.compute_cross_correlation(np.random.randn(10, 3), method="invalid_method")
    
    with pytest.raises(Exception):
        cross_section.compute_heteroscedasticity_test(np.random.randn(10), np.random.randn(10, 2), 
                                                  test_type="invalid_test")


@given(st.arrays(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e5, max_value=1e5), 
                shape=st.tuples(st.integers(10, 100), st.integers(1, 5))),
       st.arrays(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e5, max_value=1e5),
                shape=st.integers(10, 100)))
def test_property_based_regression(X, y):
    """Property-based test for regression functions using hypothesis."""
    # Skip test if arrays have incompatible dimensions
    if len(X) != len(y):
        return
    
    try:
        # Add a constant column to X
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # Run regression
        result = cross_section.cross_sectional_regression(y, X_with_const)
        
        # Check that results are produced
        assert 'params' in result
        assert 'r_squared' in result
        
        # Check that R-squared is between 0 and 1
        assert 0 <= result['r_squared'] <= 1
        
        # Check that parameters match expected dimensions
        assert len(result['params']) == X_with_const.shape[1]
        
    except Exception as e:
        # Some combinations of X and y may cause numerical issues
        # We'll allow these failures, but not others
        allowed_errors = ["singular", "underflow", "overflow", "divide by zero", "converge"]
        if not any(error in str(e).lower() for error in allowed_errors):
            raise e


@given(st.arrays(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e5, max_value=1e5),
                shape=st.tuples(st.integers(10, 100), st.integers(3, 10))))
def test_property_based_pca(X):
    """Property-based test for PCA functions using hypothesis."""
    try:
        # Run PCA
        result = cross_section.principal_component_analysis(X, standardize=True)
        
        # Check that results are produced
        assert 'eigenvectors' in result
        assert 'explained_variance_ratio' in result
        
        # Check that explained variance ratio sums to approximately 1
        assert np.isclose(sum(result['explained_variance_ratio']), 1.0, atol=1e-5)
        
        # Check that eigenvalues are non-negative
        assert all(ev >= 0 for ev in result['eigenvalues'])
        
        # Check that eigenvectors are orthogonal
        eigenvectors = result['eigenvectors']
        for i in range(eigenvectors.shape[1]):
            for j in range(i+1, eigenvectors.shape[1]):
                assert abs(np.dot(eigenvectors[:, i], eigenvectors[:, j])) < 1e-5
        
    except Exception as e:
        # Some random matrices may cause numerical issues
        # We'll allow these failures, but not others
        allowed_errors = ["singular", "underflow", "overflow", "divide by zero", "converge"]
        if not any(error in str(e).lower() for error in allowed_errors):
            raise e


@pytest.mark.asyncio
async def test_async_behavior():
    """Test asynchronous behavior if implemented in cross-section module."""
    # Generate test data
    X, y, _ = generate_regression_data(n_samples=100, n_features=3, noise=0.1)
    X_with_const = sm.add_constant(X)
    
    # Test async_regression function
    result_async = await cross_section.async_regression(y, X_with_const, robust=True)
    
    # Compare with synchronous version
    result_sync = cross_section.cross_sectional_regression(y, X_with_const, robust=True)
    
    # Results should be the same
    assert np.allclose(result_async['params'], result_sync['params'])
    
    # Test async_pca function
    X_pca, _ = generate_pca_data(n_samples=100, n_features=5, n_components=2)
    
    result_async_pca = await cross_section.async_pca(X_pca, n_components=2)
    result_sync_pca = cross_section.principal_component_analysis(X_pca, n_components=2)
    
    # Results should be the same or at least very close
    # Note: Signs of eigenvectors may be flipped, so we check absolute values
    assert np.allclose(np.abs(result_async_pca['eigenvectors']), np.abs(result_sync_pca['eigenvectors']), atol=1e-6)
    
    # Test async methods in the class-based APIs
    regression_model = cross_section.CrossSectionalRegression(robust=True)
    await regression_model.async_fit(y, X)
    
    pca_model = cross_section.PrincipalComponentAnalysis(n_components=2)
    await pca_model.async_fit(X_pca)
    
    # Verify that the models were fitted
    assert regression_model.results_ is not None
    assert pca_model.components_ is not None


def test_numba_optimization():
    """Test that Numba-optimized functions produce correct results."""
    # Test with principal_component_analysis which uses Numba optimization
    X, _ = generate_pca_data(n_samples=100, n_features=5, n_components=2)
    
    # Call the function once to trigger JIT compilation
    _ = cross_section.principal_component_analysis(X, n_components=2)
    
    # Time the function after JIT compilation
    import time
    
    start_time = time.time()
    result_optimized = cross_section.principal_component_analysis(X, n_components=2)
    optimization_time = time.time() - start_time
    
    # The function should still produce mathematically correct results
    assert 'eigenvectors' in result_optimized
    assert 'eigenvalues' in result_optimized
    assert np.isclose(sum(result_optimized['explained_variance_ratio']), 1.0, atol=1e-6)
    
    # Test other Numba-optimized functions
    X_stats = np.random.randn(100, 3)
    
    # Call once to trigger JIT compilation
    _ = cross_section.cross_sectional_stats(X_stats)
    
    # Call again to test optimized version
    result_stats = cross_section.cross_sectional_stats(X_stats)
    
    # Verify results match expected calculations
    assert np.allclose(result_stats['mean'], np.mean(X_stats, axis=0))
    assert np.allclose(result_stats['std'], np.std(X_stats, axis=0, ddof=1))