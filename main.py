"""Command-line entry point for running AI4Chem experiments."""

import argparse

from src.train import run_linear_regression_experiment, run_ridge_regression_experiment, run_random_forest_regression_experiment, run_xgboost_regression_experiment, run_svm_regression_experiment


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run AI4Chem baseline experiments.")
    parser.add_argument(
        "--model",
        choices=["linear", "ridge", "randomforest", "xgboost", "svm"],
        default="linear",
        help="Model to run.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength. Used only when --model ridge.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of estimator with Random Forest. Used only when --model randomforest or xgboost.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for XGBoost. Used only when --model xgboost.",
    )
    parser.add_argument(
        "--kernel",
        choices=["linear", "poly", "rbf", "sigmoid"],
        default="rbf",
        help="Kernel for SVM. Used only when --model svm.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter for SVM. Used only when --model svm.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon-tube parameter for SVM. Used only when --model svm.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma parameter for SVM. Used only when --model svm.",
    )
    return parser.parse_args()


def print_scores(scores: dict[str, float]) -> None:
    """Print metric scores in a readable format."""
    for metric_name, value in scores.items():
        print(f"{metric_name}: {value:.4f}")


def main() -> None:
    """Run the selected experiment."""
    args = parse_args()

    if args.model == "linear":
        scores = run_linear_regression_experiment()
    elif args.model == "ridge":
        scores = run_ridge_regression_experiment(alpha=args.alpha)
    elif args.model == "randomforest":
        scores = run_random_forest_regression_experiment(n_estimators=args.n_estimators)
    elif args.model == "xgboost":
        scores = run_xgboost_regression_experiment(n_estimators=args.n_estimators, learning_rate=args.learning_rate)
    elif args.model == "svm":
        scores = run_svm_regression_experiment(kernel=args.kernel, C=args.C, epsilon=args.epsilon)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print_scores(scores)


if __name__ == "__main__":
    main()
