"""
For now:
- Implement a simple linear regression model using the closed-form solution (Ordinary Least Squares).
- Implement a simple Ridge regression model using the closed-form solution (L2 regularization).

TO DO: 
- Implement a simple Lasso regression model using coordinate descent (L1 regularization).
- Implement a simple Elastic Net regression model using coordinate descent (combination of L1 and L2 regularization).
- And more ...
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

class LinearRegressionClosedForm:
    """A simple implementation of Ordinary Least Squares linear regression."""
    
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        """ Solve the equation X^T X w = X^T y to find the weights."""

        # Add an intercept term to X by concatenating a column of ones to the left of X
        X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))  

        # Solve for theta using the closed-form solution w = (X^T X)^(-1) X^T y
        theta = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)

        # The first element of theta is the intercept, and the rest are the weights
        self.intercept = theta[0]
        self.weights = theta[1:]

        return self.weights, self.intercept
    
    def predict(self, X):
        """Make predictions using the learned weights."""
        if self.weights is None or self.intercept is None:
            raise ValueError("Model has not been fitted yet.")
        return X @ self.weights + self.intercept


class RidgeRegressionClosedForm(LinearRegressionClosedForm):
    """A simple implementation of Ridge regression (L2 regularization)."""
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")

    def fit(self, X, y):
        """ Solve the equation (X^T X + alpha * I) w = X^T y to find the weights."""
        n_features = X.shape[1]

        # Add an intercept term to X by concatenating a column of ones to the left of X
        X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))  
        
        # Create the identity matrix for regularization, but don't regularize the intercept term
        regularization_matrix = np.eye(n_features + 1)
        regularization_matrix[0, 0] = 0  
        
        # Solve for theta using the closed-form solution w = (X^T X + alpha * I)^(-1) X^T y with regularization
        theta = np.linalg.solve(X_with_intercept.T @ X_with_intercept + self.alpha * regularization_matrix, X_with_intercept.T @ y)
        self.intercept = theta[0]
        self.weights = theta[1:]

        return self.weights, self.intercept


class RandomForest:
    """ Regression using Random Forest Estimator. """

    def __init__(self, n_estimators = 100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X_train, y_train):
        rf_model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        rf_model.fit(X_train, y_train)
        self.rf_model = rf_model
        return rf_model

    def predict(self, X):
        predictions = self.rf_model.predict(X)
        return predictions
