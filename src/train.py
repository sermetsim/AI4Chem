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

import time
import numpy as np
from src.data_utils import load_tg_csv, TRAIN_DATA_PATH, TEST_DATA_PATH, VALIDATION_DATA_PATH
from src.models import ElasticNetModel, LinearRegressionClosedForm, RidgeRegressionClosedForm, RandomForest, XGBoostModel, SupportVectorModel, create_model
from src.evaluate import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from src.featurize import psmiles_to_nparray
from src.utils import save_model
from pathlib import Path

DEFAULT_PARAM_GRIDS = {
    "ridge": [
        {"alpha": 0.001},
        {"alpha": 0.01},
        {"alpha": 0.1},
        {"alpha": 1.0},
        {"alpha": 10.0},
        {"alpha": 100.0},
        {"alpha": 1000.0}
    ],
    "randomforest": [
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 1},
        {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": 40, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 40, "min_samples_leaf": 1},
        {"n_estimators": 500, "max_depth": 40, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 3},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 3},
        {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 3},
        {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 3},
        {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 3},
        {"n_estimators": 100, "max_depth": 40, "min_samples_leaf": 3},
        {"n_estimators": 300, "max_depth": 40, "min_samples_leaf": 3},
        {"n_estimators": 500, "max_depth": 40, "min_samples_leaf": 3},
    ],
    "xgboost": [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},        
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 3},
        ],
    "svm": [
        {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 100.0, "epsilon": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 500.0, "epsilon": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 1000.0, "epsilon": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 1.0, "epsilon": 0.5, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "epsilon": 0.5, "gamma": "scale"},
        {"kernel": "rbf", "C": 100.0, "epsilon": 0.5, "gamma": "scale"},
        {"kernel": "rbf", "C": 500.0, "epsilon": 0.5, "gamma": "scale"},
        {"kernel": "rbf", "C": 1000.0, "epsilon": 0.5, "gamma": "scale"},
    ],
    "elasticnet":[
        {"alpha": 0.001, "l1_ratio": 0.1},
        {"alpha": 0.001, "l1_ratio": 0.5},
        {"alpha": 0.001, "l1_ratio": 0.9},

        {"alpha": 0.01, "l1_ratio": 0.1},
        {"alpha": 0.01, "l1_ratio": 0.5},
        {"alpha": 0.01, "l1_ratio": 0.9},

        {"alpha": 0.1, "l1_ratio": 0.1},
        {"alpha": 0.1, "l1_ratio": 0.5},
        {"alpha": 0.1, "l1_ratio": 0.9},

        {"alpha": 1.0, "l1_ratio": 0.1},
        {"alpha": 1.0, "l1_ratio": 0.5},
        {"alpha": 1.0, "l1_ratio": 0.9},
    ]
}
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

def evaluate_single_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
) -> dict[str, float]:
    """Evaluate predictions for one dataset split."""
    return {
        f"mse_{split_name}": mean_squared_error(y_true, y_pred),
        f"mae_{split_name}": mean_absolute_error(y_true, y_pred),
        f"rmse_{split_name}": root_mean_squared_error(y_true, y_pred),
        f"r2_{split_name}": r2_score(y_true, y_pred),
    }

def select_best_hyperparameters(
    model_name: str,
    param_grid: list[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[dict, float, list[dict]]:
    """Select the best hyperparameters based on validation RMSE."""
    best_params = None
    best_rmse = float("inf")
    all_results = []
    n_configs = len(param_grid)

    for config_index, params in enumerate(param_grid, start=1):
        start_time = time.perf_counter()
        model = create_model(model_name, params)
        model.fit(X_train, y_train)

        y_pred_validation = model.predict(X_validation)
        rmse_validation = root_mean_squared_error(y_validation, y_pred_validation)
        elapsed_seconds = time.perf_counter() - start_time

        result = {
            "model_name": model_name,
            "params": params,
            "rmse_validation": rmse_validation,
            "elapsed_seconds": elapsed_seconds,
        }
        all_results.append(result)

        print(
            f"[{model_name}] finished {config_index}/{n_configs} "
            f"params={params} "
            f"validation_rmse={rmse_validation:.4f} "
            f"elapsed={elapsed_seconds:.1f}s"
        )

        if rmse_validation < best_rmse:
            best_rmse = rmse_validation
            best_params = params

    return best_params, best_rmse, all_results


def retrain_best_model(
    model_name: str,
    best_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> object:
    """Retrain the best model on the merged train + validation dataset."""
    X_train_validation = np.vstack([X_train, X_validation])
    y_train_validation = np.concatenate([y_train, y_validation])

    final_model = create_model(model_name, best_params)
    final_model.fit(X_train_validation, y_train_validation)

    return final_model


def run_model_selection_experiment(
    model_name: str,
    param_grid: list[dict],
    model_output_path: Path | None = None,
) -> dict:
    """Run hyperparameter selection, retrain the best model, and evaluate on test."""
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_and_featurize_data()

    # Train and select the best hyperparameters based on validation RMSE
    best_params, best_validation_rmse, all_results = select_best_hyperparameters(
        model_name=model_name,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    # Retrain the best model on the merged train + validation dataset
    final_model = retrain_best_model(
        model_name=model_name,
        best_params=best_params,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )
    
    # Save the final model to disk as joblib file if a path is provided
    if model_output_path is not None:
        save_model(final_model, model_output_path)

    y_pred_test = final_model.predict(X_test)
    test_scores = evaluate_single_split(y_test, y_pred_test, split_name="test")

    return {
        "model_name": model_name,
        "best_params": best_params,
        "best_validation_rmse": best_validation_rmse,
        "test_scores": test_scores,
        "all_validation_results": all_results,
        "final_model": final_model,
        "model_output_path": model_output_path,
    }

## Test functions for development and debugging

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

def run_elastic_net_regression_experiment(alpha=1.0, l1_ratio=0.5) -> dict[str, float]:
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_and_featurize_data()
    model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio)
    y_pred_validation, y_pred_test = fit_and_predict(model, X_train, y_train, X_validation, X_test)
    scores = evaluate_predictions(y_validation, y_pred_validation, y_test, y_pred_test)
    return scores

def run_ensemble_regression_experiment(model_1, model_2, model_3) -> dict[str, float]:
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_and_featurize_data()
    y_pred_validation_1, y_pred_test_1 = fit_and_predict(model_1, X_train, y_train, X_validation, X_test)
    y_pred_validation_2, y_pred_test_2 = fit_and_predict(model_2, X_train, y_train, X_validation, X_test)
    y_pred_validation_3, y_pred_test_3 = fit_and_predict(model_3, X_train, y_train, X_validation, X_test)
    scores = evaluate_predictions(y_validation, (y_pred_validation_1 + y_pred_validation_2 + y_pred_validation_3) / 3, y_test, (y_pred_test_1 + y_pred_test_2 + y_pred_test_3) / 3)
    return scores
