
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

    def __init__(self, n_estimators = 100, random_state=67, max_depth=None, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rf_model = None


    def fit(self, X_train, y_train):
        rf_model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        rf_model.fit(X_train, y_train)
        self.rf_model = rf_model
        return rf_model

    def predict(self, X):
        predictions = self.rf_model.predict(X)
        return predictions


class XGBoostModel:
    """ Regression using XGBoost Estimator. """

    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_depth = max_depth
        self.xgb_model = None

    def fit(self, X_train, y_train):
        xgb_model = XGBRegressor(
            n_estimators=self.n_estimators, 
            learning_rate=self.learning_rate, 
            random_state=self.random_state,
            max_depth=self.max_depth,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        self.xgb_model = xgb_model
        return xgb_model

    def predict(self, X):
        predictions = self.xgb_model.predict(X)
        return predictions
    

class SupportVectorModel:
    """ Regression using Support Vector Machine (SVR). """

    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        self.kernel = kernel
        self.C = C # Paramètre de régularisation
        self.epsilon = epsilon # Marge de tolérance
        self.gamma = gamma # Paramètre du noyau
        self.svr_pipeline = None

    def fit(self, X_train, y_train):
        self.svr_pipeline = make_pipeline(StandardScaler(), SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma))
        self.svr_pipeline.fit(X_train, y_train)
        return self.svr_pipeline

    def predict(self, X):
        predictions = self.svr_pipeline.predict(X)
        return predictions
    
class ElasticNetModel:
    """ Regression using Elastic Net. """

    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.elastic_net = None

    def fit(self, X_train, y_train):
        self.elastic_net = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        self.elastic_net.fit(X_train, y_train)
        return self.elastic_net

    def predict(self, X):
        predictions = self.elastic_net.predict(X)
        return predictions 
    

def create_model(model_name: str, params: dict) -> object:
    """Factory function to create a model instance based on the model name and parameters."""
    model_classes = {
        "linear": LinearRegressionClosedForm,
        "ridge": RidgeRegressionClosedForm,
        "randomforest": RandomForest,
        "xgboost": XGBoostModel,
        "svm": SupportVectorModel,
        "elasticnet": ElasticNetModel,
    }

    if model_name not in model_classes:
        raise ValueError(f"Unknown model name: {model_name}")

    model_class = model_classes[model_name]
    return model_class(**params)
