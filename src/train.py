"""
For now:
- Train and solve using a simple linear regression model using the closed-form solution (Ordinary Least Squares).
- Train and solve using a simple Ridge regression model using the closed-form solution (L2 regularization).
- Evaluate the models using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R^2 Score on the validation and test sets.
TO DO:
- Train and solve using a simple Lasso regression model using coordinate descent (L1 regularization
- Train and solve using a simple Elastic Net regression model using coordinate descent (combination of L1 and L2 regularization).
- And more ...
"""

import numpy as np
from src.data_utils import load_tg_csv, TRAIN_DATA_PATH, TEST_DATA_PATH, VALIDATION_DATA_PATH
from src.models import LinearRegressionClosedForm, RidgeRegressionClosedForm, RandomForest, XGBoostModel, SupportVectorModel
from src.evaluate import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from src.featurize import psmiles_to_nparray
from pathlib import Path


def load_and_featurize_data(train_path: Path = TRAIN_DATA_PATH, 
                            test_path: Path = TEST_DATA_PATH, 
                            validation_path: Path = VALIDATION_DATA_PATH
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the datasets and featurize the PSMILES strings into numpy arrays of Morgan fingerprints."""

    # Load the datasets
    X_train, y_train = load_tg_csv(train_path)
    X_test, y_test = load_tg_csv(test_path)
    X_validation, y_validation = load_tg_csv(validation_path)

    # Transform the data as needed (e.g., convert lists to numpy arrays)
    # using the featurization with Morgan fingerprints
    X_train = psmiles_to_nparray(X_train)
    X_validation = psmiles_to_nparray(X_validation)
    X_test = psmiles_to_nparray(X_test)

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def evaluate_predictions(y_validation: np.ndarray, 
                        y_pred_validation: np.ndarray,
                        y_test: np.ndarray, 
                        y_pred_test: np.ndarray
                        ) -> dict[str, float]:
    """Evaluate the predictions on the validation and test sets using various metrics and return the scores in a dictionary."""

    scores = {
        "mse_validation": mean_squared_error(y_validation, y_pred_validation),
        "mse_test": mean_squared_error(y_test, y_pred_test),
        "mae_validation": mean_absolute_error(y_validation, y_pred_validation),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "rmse_validation": root_mean_squared_error(y_validation, y_pred_validation),
        "rmse_test": root_mean_squared_error(y_test, y_pred_test),
        "r2_validation": r2_score(y_validation, y_pred_validation),
        "r2_test": r2_score(y_test, y_pred_test)
    }
    return scores

def fit_and_predict(model: object, 
                    X_train: np.ndarray, 
                    y_train: np.ndarray, 
                    X_validation: np.ndarray, 
                    X_test: np.ndarray, 
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Train the given model on the training data and make predictions on the validation and test sets."""

    # Initialize and fit the linear regression model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_validation = model.predict(X_validation)
    y_pred_test = model.predict(X_test)

    return y_pred_validation, y_pred_test

def run_linear_regression_experiment() -> dict[str, float]:
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_and_featurize_data()
    model = LinearRegressionClosedForm()
    y_pred_validation, y_pred_test = fit_and_predict(model, X_train, y_train, X_validation, X_test)
    scores = evaluate_predictions(y_validation, y_pred_validation, y_test, y_pred_test)
    return scores

def run_ridge_regression_experiment(alpha: float = 1.0) -> dict[str, float]:
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_and_featurize_data()
    model = RidgeRegressionClosedForm(alpha=alpha)
    y_pred_validation, y_pred_test = fit_and_predict(model, X_train, y_train, X_validation, X_test)
    scores = evaluate_predictions(y_validation, y_pred_validation, y_test, y_pred_test)
    return scores

def run_random_forest_regression_experiment(n_estimators = 100) -> dict[str, float]:
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_and_featurize_data()
    model = RandomForest(n_estimators=n_estimators)
    y_pred_validation, y_pred_test = fit_and_predict(model, X_train, y_train, X_validation, X_test)
    scores = evaluate_predictions(y_validation, y_pred_validation, y_test, y_pred_test)
    return scores

def run_xgboost_regression_experiment(n_estimators=100, learning_rate=0.1) -> dict[str, float]:
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_and_featurize_data()
    model = XGBoostModel(n_estimators=n_estimators, learning_rate=learning_rate)
    y_pred_validation, y_pred_test = fit_and_predict(model, X_train, y_train, X_validation, X_test)
    scores = evaluate_predictions(y_validation, y_pred_validation, y_test, y_pred_test)
    return scores

def run_svm_regression_experiment(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale') -> dict[str, float]:
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_and_featurize_data()
    model = SupportVectorModel(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    y_pred_validation, y_pred_test = fit_and_predict(model, X_train, y_train, X_validation, X_test)
    scores = evaluate_predictions(y_validation, y_pred_validation, y_test, y_pred_test)
    return scores