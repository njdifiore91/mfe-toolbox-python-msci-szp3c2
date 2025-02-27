"""
Comprehensive test suite for the MFE Toolbox validation module.

This module tests all functions in the validation module, ensuring proper 
input validation, type checking, and error handling across the toolbox's 
validation framework.
"""

import pytest  # pytest 7.4.3
import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
from hypothesis import given, strategies as st  # hypothesis 6.92.1
import scipy.linalg  # scipy 1.11.4
from typing import Any
import numbers  # Python 3.12

# Import the validation functions to be tested
from mfe.utils.validation import (
    validate_array, validate_data, validate_params, check_dimension, 
    check_dimensions, is_positive_integer, is_non_negative_integer,
    is_positive_float, is_probability, check_lags, check_range,
    check_order, check_time_series, is_float_array, is_int_array,
    is_positive_definite, is_symmetric, check_strictly_increasing,
    check_shape_compatibility, validate_type
)

# Import test utilities
from . import create_test_samples, validate_test_data
from .. import check_test_environment, prepare_test_data

# Mark all tests in this module with 'utils' marker
pytestmark = [pytest.mark.utils]


def test_validate_array_basic():
    """Tests basic functionality of validate_array with valid inputs"""
    # Create test array
    arr = np.array([1, 2, 3, 4, 5])
    
    # Test with default parameters
    result = validate_array(arr)
    assert np.array_equal(result, arr)
    
    # Test with expected shape
    result = validate_array(arr, expected_shape=(5,))
    assert np.array_equal(result, arr)
    
    # Test with expected dtype
    result = validate_array(arr, expected_dtype=np.float64)
    assert result.dtype == np.float64
    assert np.array_equal(result, arr.astype(np.float64))


def test_validate_array_errors():
    """Tests that validate_array properly raises errors for invalid inputs"""
    # Test with non-array input that can't be converted
    with pytest.raises(TypeError):
        validate_array(lambda x: x, param_name="lambda_func")
    
    # Test with wrong shape
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        validate_array(arr, expected_shape=(4,))
    
    # Test with incompatible dtype
    arr = np.array(['a', 'b', 'c'])
    with pytest.raises(TypeError):
        validate_array(arr, expected_dtype=np.float64)
    
    # Test with None and allow_none=False
    with pytest.raises(TypeError):
        validate_array(None, allow_none=False)


def test_validate_data_conversion():
    """Tests that validate_data properly converts various input types"""
    # Test with numpy array
    arr = np.array([1, 2, 3, 4, 5])
    result = validate_data(arr)
    assert np.array_equal(result, arr)
    
    # Test with pandas Series
    series = pd.Series([1, 2, 3, 4, 5])
    result = validate_data(series)
    assert np.array_equal(result, series.values)
    
    # Test with pandas DataFrame
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    result = validate_data(df)
    assert np.array_equal(result, df.values)
    
    # Test with list
    lst = [1, 2, 3, 4, 5]
    result = validate_data(lst)
    assert np.array_equal(result, np.array(lst))


def test_validate_data_parameters():
    """Tests validate_data with various parameter combinations"""
    # Test with min_length
    arr = np.array([1, 2, 3, 4, 5])
    result = validate_data(arr, min_length=3)
    assert np.array_equal(result, arr)
    
    # Test with ensure_2d
    result = validate_data(arr, ensure_2d=True)
    assert result.shape == (5, 1)
    
    # Test with ensure_contiguous
    non_contiguous = np.array([[1, 2], [3, 4]])[::2, ::2]
    result = validate_data(non_contiguous, ensure_contiguous=True)
    assert result.flags.c_contiguous
    
    # Test with dtype
    result = validate_data(arr, dtype=np.float64)
    assert result.dtype == np.float64


def test_validate_data_errors():
    """Tests that validate_data raises appropriate errors for invalid inputs"""
    # Test with input shorter than min_length
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        validate_data(arr, min_length=5)
    
    # Test with None without allow_none
    with pytest.raises(TypeError):
        validate_data(None, allow_none=False)
    
    # Test with invalid input type
    with pytest.raises(TypeError):
        validate_data(lambda x: x)
    
    # Test with incompatible dtype conversion
    arr = np.array(['a', 'b', 'c'])
    with pytest.raises(TypeError):
        validate_data(arr, dtype=np.float64)


def test_validate_params_basic():
    """Tests basic functionality of validate_params with valid inputs"""
    # Create parameter dictionary
    params = {
        'a': 10,
        'b': 'string',
        'c': 0.5
    }
    
    # Create parameter specifications
    param_specs = {
        'a': {'type': int, 'min': 0, 'max': 100},
        'b': {'type': str},
        'c': {'type': float, 'min': 0, 'max': 1}
    }
    
    # Test validation
    assert validate_params(params, param_specs)
    
    # Test with optional parameter missing
    params_missing_optional = {'a': 10, 'c': 0.5}
    param_specs_with_optional = {
        'a': {'type': int, 'required': True},
        'b': {'type': str, 'required': False},
        'c': {'type': float, 'required': True}
    }
    assert validate_params(params_missing_optional, param_specs_with_optional)


def test_validate_params_errors():
    """Tests that validate_params correctly identifies invalid parameters"""
    # Test with missing required parameter
    params = {'a': 10}
    param_specs = {
        'a': {'type': int, 'required': True},
        'b': {'type': str, 'required': True}
    }
    assert not validate_params(params, param_specs, raise_error=False)
    with pytest.raises(ValueError):
        validate_params(params, param_specs)
    
    # Test with parameter of wrong type
    params = {'a': 10, 'b': 123}
    param_specs = {
        'a': {'type': int},
        'b': {'type': str}
    }
    assert not validate_params(params, param_specs, raise_error=False)
    with pytest.raises(TypeError):
        validate_params(params, param_specs)
    
    # Test with value outside constraints
    params = {'a': 10, 'b': 'string', 'c': 1.5}
    param_specs = {
        'a': {'type': int},
        'b': {'type': str},
        'c': {'type': float, 'max': 1.0}
    }
    assert not validate_params(params, param_specs, raise_error=False)
    with pytest.raises(ValueError):
        validate_params(params, param_specs)
    
    # Test with extra parameters when strict=True
    params = {'a': 10, 'b': 'string', 'extra': 123}
    param_specs = {
        'a': {'type': int},
        'b': {'type': str}
    }
    assert not validate_params(params, param_specs, strict=True, raise_error=False)
    with pytest.raises(ValueError):
        validate_params(params, param_specs, strict=True)


def test_check_dimension():
    """Tests check_dimension function for array dimension validation"""
    # Create arrays with different dimensions
    arr_1d = np.array([1, 2, 3])
    arr_2d = np.array([[1, 2], [3, 4]])
    arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    
    # Test with correct dimensions
    assert check_dimension(arr_1d, 1)
    assert check_dimension(arr_2d, 2)
    assert check_dimension(arr_3d, 3)
    
    # Test with incorrect dimensions
    assert not check_dimension(arr_1d, 2, raise_error=False)
    assert not check_dimension(arr_2d, 1, raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        check_dimension(arr_1d, 2)
    with pytest.raises(ValueError):
        check_dimension(arr_2d, 3)
    
    # Test with non-array input
    with pytest.raises(TypeError):
        check_dimension([1, 2, 3], 1)


def test_check_dimensions():
    """Tests check_dimensions function for validating multiple allowed dimensions"""
    # Create arrays with different dimensions
    arr_1d = np.array([1, 2, 3])
    arr_2d = np.array([[1, 2], [3, 4]])
    arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    
    # Test with list of allowed dimensions
    assert check_dimensions(arr_1d, [1, 2])
    assert check_dimensions(arr_2d, [2, 3])
    assert not check_dimensions(arr_3d, [1, 2], raise_error=False)
    
    # Test with single dimension as integer
    assert check_dimensions(arr_1d, 1)
    assert check_dimensions(arr_2d, 2)
    assert not check_dimensions(arr_1d, 2, raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        check_dimensions(arr_3d, [1, 2])
    
    # Test with non-array input
    with pytest.raises(TypeError):
        check_dimensions([1, 2, 3], [1, 2])


def test_is_positive_integer():
    """Tests is_positive_integer function for positive integer validation"""
    # Test with positive integers
    assert is_positive_integer(1)
    assert is_positive_integer(np.int32(10))
    assert is_positive_integer(100)
    
    # Test with zero and negative integers
    assert not is_positive_integer(0, raise_error=False)
    assert not is_positive_integer(-1, raise_error=False)
    
    # Test with non-integers
    assert not is_positive_integer(1.5, raise_error=False)
    assert not is_positive_integer("1", raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        is_positive_integer(0)
    with pytest.raises(ValueError):
        is_positive_integer(-5)
    with pytest.raises(ValueError):
        is_positive_integer(1.5)


def test_is_non_negative_integer():
    """Tests is_non_negative_integer function for non-negative integer validation"""
    # Test with positive and zero integers
    assert is_non_negative_integer(0)
    assert is_non_negative_integer(1)
    assert is_non_negative_integer(np.int32(10))
    
    # Test with negative integers
    assert not is_non_negative_integer(-1, raise_error=False)
    assert not is_non_negative_integer(-100, raise_error=False)
    
    # Test with non-integers
    assert not is_non_negative_integer(1.5, raise_error=False)
    assert not is_non_negative_integer("0", raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        is_non_negative_integer(-5)
    with pytest.raises(ValueError):
        is_non_negative_integer(1.5)


def test_is_positive_float():
    """Tests is_positive_float function for positive number validation"""
    # Test with positive numbers
    assert is_positive_float(1.0)
    assert is_positive_float(1)  # Integer is also valid
    assert is_positive_float(np.float32(0.5))
    
    # Test with zero
    assert not is_positive_float(0.0, zero_allowed=False, raise_error=False)
    assert is_positive_float(0.0, zero_allowed=True)
    
    # Test with negative numbers
    assert not is_positive_float(-1.0, raise_error=False)
    assert not is_positive_float(-0.1, raise_error=False)
    
    # Test with non-numeric types
    assert not is_positive_float("1.0", raise_error=False)
    assert not is_positive_float([1.0], raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        is_positive_float(-1.0)
    with pytest.raises(ValueError):
        is_positive_float(0.0, zero_allowed=False)
    with pytest.raises(ValueError):
        is_positive_float("1.0")


def test_is_probability():
    """Tests is_probability function for probability validation (0-1 range)"""
    # Test with values in range [0, 1]
    assert is_probability(0.0)
    assert is_probability(0.5)
    assert is_probability(1.0)
    assert is_probability(np.float32(0.75))
    
    # Test with values outside range
    assert not is_probability(-0.1, raise_error=False)
    assert not is_probability(1.1, raise_error=False)
    
    # Test with non-numeric types
    assert not is_probability("0.5", raise_error=False)
    assert not is_probability([0.5], raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        is_probability(-0.1)
    with pytest.raises(ValueError):
        is_probability(1.1)
    with pytest.raises(ValueError):
        is_probability("0.5")


def test_check_lags():
    """Tests check_lags function for validating lag specifications"""
    # Test with single integer
    lags = check_lags(5)
    assert np.array_equal(lags, np.array([5]))
    
    # Test with list of lags
    lags = check_lags([1, 3, 5])
    assert np.array_equal(lags, np.array([1, 3, 5]))
    
    # Test with numpy array
    lags = check_lags(np.array([2, 4, 6]))
    assert np.array_equal(lags, np.array([2, 4, 6]))
    
    # Test with max_lag
    lags = check_lags([1, 3, 5], max_lag=5)
    assert np.array_equal(lags, np.array([1, 3, 5]))
    
    # Test with unsorted lags (should be sorted)
    lags = check_lags([5, 1, 3])
    assert np.array_equal(lags, np.array([1, 3, 5]))
    
    # Test with negative lags
    with pytest.raises(ValueError):
        check_lags([-1, 2, 3])
    
    # Test with lags exceeding max_lag
    with pytest.raises(ValueError):
        check_lags([1, 5, 10], max_lag=7)


def test_check_range():
    """Tests check_range function for validating values within specified ranges"""
    # Test with value inside range
    assert check_range(5, 0, 10)
    assert check_range(0.5, 0, 1)
    
    # Test with inclusive boundaries
    assert check_range(0, 0, 10, inclusive_min=True)
    assert check_range(10, 0, 10, inclusive_max=True)
    
    # Test with exclusive boundaries
    assert not check_range(0, 0, 10, inclusive_min=False, raise_error=False)
    assert not check_range(10, 0, 10, inclusive_max=False, raise_error=False)
    
    # Test with value outside range
    assert not check_range(-1, 0, 10, raise_error=False)
    assert not check_range(11, 0, 10, raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        check_range(-1, 0, 10)
    with pytest.raises(ValueError):
        check_range(11, 0, 10)
    with pytest.raises(ValueError):
        check_range(0, 0, 10, inclusive_min=False)
    with pytest.raises(ValueError):
        check_range(10, 0, 10, inclusive_max=False)


def test_check_order():
    """Tests check_order function for validating model order specifications"""
    # Test with valid ARIMA order
    assert check_order((1, 0, 1), 'ARIMA')
    
    # Test with valid GARCH order
    assert check_order((1, 1), 'GARCH')
    
    # Test with incorrect length for model type
    assert not check_order((1, 1), 'ARIMA', raise_error=False)
    assert not check_order((1, 0, 1), 'GARCH', raise_error=False)
    
    # Test with negative values
    assert not check_order((1, -1, 1), 'ARIMA', raise_error=False)
    assert not check_order((-1, 1), 'GARCH', raise_error=False)
    
    # Test with non-integer values
    assert not check_order((1.5, 0, 1), 'ARIMA', raise_error=False)
    assert not check_order((1, 0.5), 'GARCH', raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        check_order((1, 1), 'ARIMA')
    with pytest.raises(ValueError):
        check_order((1, 0, 1), 'GARCH')
    with pytest.raises(ValueError):
        check_order((1, -1, 1), 'ARIMA')
    with pytest.raises(ValueError):
        check_order((1.5, 0, 1), 'ARIMA')


def test_check_time_series():
    """Tests check_time_series function for validating time series data"""
    # Test with valid numpy array
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = check_time_series(arr)
    assert np.array_equal(result, arr)
    
    # Test with valid pandas Series
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = check_time_series(series)
    assert np.array_equal(result, series.values)
    
    # Test with valid list
    lst = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = check_time_series(lst)
    assert np.array_equal(result, np.array(lst))
    
    # Test with min_length
    result = check_time_series(arr, min_length=3)
    assert np.array_equal(result, arr)
    with pytest.raises(ValueError):
        check_time_series(arr, min_length=10)
    
    # Test with allow_2d
    arr_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(ValueError):
        check_time_series(arr_2d, allow_2d=False)
    result = check_time_series(arr_2d, allow_2d=True)
    assert np.array_equal(result, arr_2d)
    
    # Test with allow_missing
    arr_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    with pytest.raises(ValueError):
        check_time_series(arr_with_nan, allow_missing=False)
    result = check_time_series(arr_with_nan, allow_missing=True)
    assert np.array_equal(result, arr_with_nan, equal_nan=True)


def test_is_float_array():
    """Tests is_float_array function for floating-point array validation"""
    # Test with float arrays
    assert is_float_array(np.array([1.0, 2.0, 3.0]))
    assert is_float_array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    
    # Test with integer arrays
    assert not is_float_array(np.array([1, 2, 3]), raise_error=False)
    
    # Test with mixed type arrays
    mixed_arr = np.array([1, 2.0, 3])
    assert not is_float_array(mixed_arr, raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(TypeError):
        is_float_array(np.array([1, 2, 3]))
    with pytest.raises(TypeError):
        is_float_array([1.0, 2.0, 3.0])  # Not a numpy array


def test_is_int_array():
    """Tests is_int_array function for integer array validation"""
    # Test with integer arrays
    assert is_int_array(np.array([1, 2, 3]))
    assert is_int_array(np.array([1, 2, 3], dtype=np.int32))
    
    # Test with float arrays
    assert not is_int_array(np.array([1.0, 2.0, 3.0]), raise_error=False)
    
    # Test with mixed type arrays
    mixed_arr = np.array([1, 2.0, 3])
    assert not is_int_array(mixed_arr, raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(TypeError):
        is_int_array(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError):
        is_int_array([1, 2, 3])  # Not a numpy array


def test_is_positive_definite():
    """Tests is_positive_definite function for validating positive definite matrices"""
    # Create a positive definite matrix
    pd_matrix = np.array([[2, 1], [1, 2]])
    assert is_positive_definite(pd_matrix)
    
    # Create a non-positive definite matrix
    non_pd_matrix = np.array([[1, 2], [2, 1]])
    assert not is_positive_definite(non_pd_matrix, raise_error=False)
    
    # Test with non-square matrix
    non_square = np.array([[1, 2, 3], [4, 5, 6]])
    assert not is_positive_definite(non_square, raise_error=False)
    
    # Test with non-matrix input
    with pytest.raises(ValueError):
        is_positive_definite([1, 2, 3])
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        is_positive_definite(non_pd_matrix)
    with pytest.raises(ValueError):
        is_positive_definite(non_square)


def test_is_symmetric():
    """Tests is_symmetric function for validating symmetric matrices"""
    # Create a symmetric matrix
    sym_matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    assert is_symmetric(sym_matrix)
    
    # Create a non-symmetric matrix
    non_sym_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert not is_symmetric(non_sym_matrix, raise_error=False)
    
    # Test with different tolerance values
    almost_sym = np.array([[1, 2.000001], [2, 3]])
    assert is_symmetric(almost_sym, tolerance=1e-5)
    assert not is_symmetric(almost_sym, tolerance=1e-10, raise_error=False)
    
    # Test with non-square matrix
    non_square = np.array([[1, 2, 3], [4, 5, 6]])
    assert not is_symmetric(non_square, raise_error=False)
    
    # Test with non-matrix input
    with pytest.raises(ValueError):
        is_symmetric([1, 2, 3])
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        is_symmetric(non_sym_matrix)
    with pytest.raises(ValueError):
        is_symmetric(non_square)


def test_check_strictly_increasing():
    """Tests check_strictly_increasing for validating monotonically increasing arrays"""
    # Create a strictly increasing array
    inc_arr = np.array([1, 2, 3, 4, 5])
    assert check_strictly_increasing(inc_arr)
    
    # Create arrays with plateaus and decreasing segments
    plateau_arr = np.array([1, 2, 2, 3, 4])
    decreasing_arr = np.array([1, 2, 3, 2, 4])
    
    assert not check_strictly_increasing(plateau_arr, raise_error=False)
    assert not check_strictly_increasing(decreasing_arr, raise_error=False)
    
    # Test with single element array
    single_element = np.array([1])
    assert check_strictly_increasing(single_element)
    
    # Test with empty array
    empty_arr = np.array([])
    assert check_strictly_increasing(empty_arr)
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        check_strictly_increasing(plateau_arr)
    with pytest.raises(ValueError):
        check_strictly_increasing(decreasing_arr)
    with pytest.raises(ValueError):
        check_strictly_increasing([1, 2, 3])  # Not a numpy array


def test_check_shape_compatibility():
    """Tests check_shape_compatibility for validating array shape compatibility"""
    # Create arrays with compatible shapes for addition
    a1 = np.array([[1, 2], [3, 4]])
    a2 = np.array([[5, 6], [7, 8]])
    assert check_shape_compatibility([a1, a2], 'addition')
    
    # Create arrays with compatible shapes for multiplication
    b1 = np.array([[1, 2], [3, 4]])
    b2 = np.array([[5, 6], [7, 8]])
    assert check_shape_compatibility([b1, b2], 'multiplication')
    
    # Create arrays with compatible shapes for broadcasting
    c1 = np.array([[1, 2], [3, 4]])
    c2 = np.array([5, 6])
    assert check_shape_compatibility([c1, c2], 'broadcasting')
    
    # Create arrays with incompatible shapes
    d1 = np.array([[1, 2], [3, 4]])
    d2 = np.array([[5, 6, 7], [8, 9, 10]])
    
    assert not check_shape_compatibility([d1, d2], 'addition', raise_error=False)
    assert not check_shape_compatibility([d1, d2], 'multiplication', raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(ValueError):
        check_shape_compatibility([d1, d2], 'addition')
    with pytest.raises(ValueError):
        check_shape_compatibility([a1], 'multiplication')  # Need exactly 2 arrays
    with pytest.raises(ValueError):
        check_shape_compatibility([a1, a2], 'unknown_operation')


def test_validate_type():
    """Tests validate_type function for type validation"""
    # Test with correct types
    assert validate_type(10, int)
    assert validate_type(10.5, float)
    assert validate_type("string", str)
    assert validate_type([1, 2, 3], list)
    
    # Test with multiple allowed types
    assert validate_type(10, (int, float))
    assert validate_type(10.5, (int, float))
    assert validate_type(None, (int, str, type(None)))
    
    # Test with incorrect types
    assert not validate_type(10, str, raise_error=False)
    assert not validate_type("10", int, raise_error=False)
    assert not validate_type(None, int, raise_error=False)
    
    # Test with raise_error=True
    with pytest.raises(TypeError):
        validate_type(10, str)
    with pytest.raises(TypeError):
        validate_type("10", int)
    with pytest.raises(TypeError):
        validate_type(None, int)


@given(arrays=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100).map(np.array))
def test_validate_array_property(arrays):
    """Property-based test for validate_array using hypothesis"""
    # Test with default parameters
    result = validate_array(arrays)
    assert np.array_equal(result, arrays)
    
    # Test with correct shape
    shape = arrays.shape
    result = validate_array(arrays, expected_shape=shape)
    assert result.shape == shape
    
    # Test with correct dtype
    dtype = arrays.dtype
    result = validate_array(arrays, expected_dtype=dtype)
    assert result.dtype == dtype


@given(value=st.floats(allow_nan=False, allow_infinity=False), 
       min_value=st.floats(max_value=0), 
       max_value=st.floats(min_value=1))
def test_check_range_property(value, min_value, max_value):
    """Property-based test for check_range using hypothesis"""
    # Ensure min_value < max_value
    min_value = min(min_value, max_value - 1)
    max_value = max(max_value, min_value + 1)
    
    # Check if value is within range
    if min_value <= value <= max_value:
        assert check_range(value, min_value, max_value)
    else:
        assert not check_range(value, min_value, max_value, raise_error=False)


@given(params=st.dictionaries(
    keys=st.text(min_size=1),
    values=st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text())
))
def test_validate_params_property(params):
    """Property-based test for validate_params using hypothesis"""
    # Create parameter specifications based on the parameters
    param_specs = {}
    for key, value in params.items():
        param_specs[key] = {'type': type(value), 'required': True}
    
    # Test validation
    assert validate_params(params, param_specs)