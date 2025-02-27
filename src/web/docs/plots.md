# Plotting in MFE Toolbox

Comprehensive documentation of plotting capabilities in the MFE Toolbox, explaining the matplotlib-based visualization components, their integration with PyQt6, asynchronous plot updates, and usage patterns for econometric analysis visualizations.

## Introduction

The MFE Toolbox provides a comprehensive plotting system for visualizing econometric analysis results. It leverages the power of `matplotlib` <!-- matplotlib 3.7.1+ --> for creating high-quality plots and integrates seamlessly with `PyQt6` <!-- PyQt6 6.6.1+ --> for interactive GUI applications. This documentation covers the core architecture, available plot components, customization options, and integration examples.

## Plot Architecture

The plotting system is built around a modular architecture that separates the plotting logic from the UI components. The core components are:

- **MatplotlibBackend**: Manages the `matplotlib` <!-- matplotlib 3.7.1+ --> figure and canvas, and handles the integration with `PyQt6` <!-- PyQt6 6.6.1+ -->.
- **AsyncPlotWidget**: A base class for all plot widgets, providing support for asynchronous plot updates.
- **Plot Components**: Specialized widgets for visualizing specific types of data, such as time series, residuals, and volatility models.

## Time Series Visualization

The `TimeSeriesPlot` component is used for visualizing time series data, including observed data, fitted values, forecasts, and confidence intervals.

```python
# Create time series plot widget
ts_plot = TimeSeriesPlot()

# Set observed data
ts_plot.set_data(returns_data, dates=date_index)

# Add model fit and forecast
ts_plot.set_fitted_values(model_results.fitted_values)
ts_plot.set_forecast(model_results.forecast, 
                     lower_ci=model_results.forecast_lower_ci,
                     upper_ci=model_results.forecast_upper_ci)

# Update the plot
await ts_plot.async_update_plot()
```

## Residual Diagnostics

The `ResidualPlot` component provides a comprehensive view of residual diagnostics, including a time series plot, histogram, and Q-Q plot.

```python
# Create residual plot widget
res_plot = ResidualPlot()

# Set residual data from model estimation
res_plot.set_residuals(model_results.residuals)

# Customize appearance
res_plot.set_plot_params({
    'title': 'Model Residuals',
    'hist_bins': 25,
    'qqplot': True
})

# Update the plot
await res_plot.async_update_plot()
```

## Autocorrelation Plots

The `ACFPlot` and `PACFPlot` components are used for visualizing the autocorrelation and partial autocorrelation functions of a time series.

```python
# Create ACF plot widget
acf_plot = ACFPlot()

# Set data from model residuals
acf_plot.set_data(model_results.residuals, max_lags=30, alpha=0.05)

# Update the plot
acf_plot.update_plot()
```

## Volatility Visualization

The `VolatilityPlot` component is used for visualizing volatility model results, including conditional variance and news impact curves.

```python
# Create volatility plot widget
vol_plot = VolatilityPlot()

# Set returns and conditional variance
vol_plot.set_returns(returns_data)
vol_plot.set_conditional_variance(garch_results.conditional_variance)

# Add news impact curve
vol_plot.set_news_impact_curve(garch_results.news_impact)

# Update the plot
await vol_plot.async_update_plot()
```

## Distribution Analysis

The `QQPlot` component is used for quantile-quantile plots, comparing the empirical distribution of data against theoretical distributions.

```python
# Create Q-Q plot widget
qq_plot = QQPlot()

# Set data comparing against t-distribution
qq_plot.set_data(model_results.residuals, dist='t', df=5)

# Update the plot
qq_plot.update_plot()