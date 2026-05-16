"""
Simple evaluation metrics for regression models, including Mean Squared Error (MSE), 
Mean Absolute Error (MAE), and R^2 Score.
"""

import numpy as np

def mean_squared_error(y_true, y_pred):
    """Calculate the Mean Squared Error between true and predicted values."""
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """Calculate the Root Mean Squared Error between true and predicted values."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    """Calculate the Mean Absolute Error between true and predicted values."""
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """Calculate the R^2 (coefficient of determination) score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0