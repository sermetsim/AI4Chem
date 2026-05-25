"""Command-line entry point for running AI4Chem experiments."""

import argparse
from pathlib import Path

from src.train import (
    DEFAULT_PARAM_GRIDS,
    evaluate_single_split,
    load_and_featurize_data,
    run_model_selection_experiment,
)
from src.utils import flatten_result_row, load_model, save_dicts_to_csv


MODEL_DIR = Path("artifacts") / "models"
RESULTS_DIR = Path("artifacts") / "results"


def default_model_path(model_name: str) -> Path:
    """Return the default checkpoint path for a model name."""
    return MODEL_DIR / f"best_{model_name}.joblib"


def default_results_path(model_name: str) -> Path:
    """Return the default summary results path for a model name."""
    return RESULTS_DIR / model_name / f"{model_name}_summary.csv"


def default_validation_results_path(model_name: str) -> Path:
    """Return the default validation sweep results path for a model name."""
    return RESULTS_DIR / model_name / f"{model_name}_validation_sweep.csv"


def default_test_predictions_path(model_name: str) -> Path:
    """Return the default test predictions path for a model name."""
    return RESULTS_DIR / model_name / f"{model_name}_test_predictions.csv"


def default_test_metrics_path(model_name: str) -> Path:
    """Return the default test metrics path for a model name."""
    return RESULTS_DIR / model_name / f"{model_name}_test_metrics.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run AI4Chem experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Select hyperparameters, retrain on train + validation, and save the model.",
    )
    train_parser.add_argument(
        "--model",
        choices=sorted(DEFAULT_PARAM_GRIDS),
        required=True,
        help="Model to train.",
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path where the trained model checkpoint will be saved.",
    )
    train_parser.add_argument(
        "--results-output",
        type=Path,
        default=None,
        help="Path where the final summary CSV will be saved.",
    )
    train_parser.add_argument(
        "--validation-output",
        type=Path,
        default=None,
        help="Path where the validation sweep CSV will be saved.",
    )

    test_parser = subparsers.add_parser(
        "test",
        help="Load a saved model checkpoint and evaluate it on the test set.",
    )
    test_parser.add_argument(
        "--model",
        choices=sorted(DEFAULT_PARAM_GRIDS),
        required=True,
        help="Model checkpoint to load if --checkpoint is not provided.",
    )
    test_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a saved model checkpoint.",
    )
    test_parser.add_argument(
        "--predictions-output",
        type=Path,
        default=None,
        help="Path where test predictions CSV will be saved.",
    )
    test_parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Path where test metrics CSV will be saved.",
    )

    return parser.parse_args()


def print_scores(scores: dict[str, float]) -> None:
    """Print metric scores in a readable format."""
    for metric_name, value in scores.items():
        print(f"{metric_name}: {value:.4f}")


def print_train_results(results: dict) -> None:
    """Print model-selection and final test results."""
    print(f"model_name: {results['model_name']}")
    print(f"best_params: {results['best_params']}")
    print(f"best_validation_rmse: {results['best_validation_rmse']:.4f}")
    print(f"model_output_path: {results['model_output_path']}")
    print_scores(results["test_scores"])


def save_train_results(results: dict, summary_path: Path, validation_path: Path) -> None:
    """Save final summary and validation sweep results to CSV files."""
    summary_row = {
        "model_name": results["model_name"],
        "best_params": results["best_params"],
        "best_validation_rmse": results["best_validation_rmse"],
        "model_output_path": results["model_output_path"],
        **results["test_scores"],
    }
    save_dicts_to_csv([flatten_result_row(summary_row)], summary_path)
    save_dicts_to_csv(
        [flatten_result_row(row) for row in results["all_validation_results"]],
        validation_path,
    )


def run_train_command(args: argparse.Namespace) -> None:
    """Run the train command."""
    output_path = args.output if args.output is not None else default_model_path(args.model)
    summary_path = (
        args.results_output
        if args.results_output is not None
        else default_results_path(args.model)
    )
    validation_path = (
        args.validation_output
        if args.validation_output is not None
        else default_validation_results_path(args.model)
    )

    results = run_model_selection_experiment(
        model_name=args.model,
        param_grid=DEFAULT_PARAM_GRIDS[args.model],
        model_output_path=output_path,
    )
    save_train_results(results, summary_path, validation_path)
    print_train_results(results)
    print(f"summary_results_path: {summary_path}")
    print(f"validation_results_path: {validation_path}")


def run_test_command(args: argparse.Namespace) -> None:
    """Run the test command."""
    checkpoint_path = args.checkpoint if args.checkpoint is not None else default_model_path(args.model)
    predictions_path = (
        args.predictions_output
        if args.predictions_output is not None
        else default_test_predictions_path(args.model)
    )
    metrics_path = (
        args.metrics_output
        if args.metrics_output is not None
        else default_test_metrics_path(args.model)
    )
    model = load_model(checkpoint_path)

    _, _, _, _, X_test, y_test = load_and_featurize_data()
    y_pred_test = model.predict(X_test)
    test_scores = evaluate_single_split(y_test, y_pred_test, split_name="test")
    prediction_rows = [
        {"sample_index": index, "y_true": y_true, "y_pred": y_pred}
        for index, (y_true, y_pred) in enumerate(zip(y_test, y_pred_test))
    ]

    save_dicts_to_csv(prediction_rows, predictions_path)
    save_dicts_to_csv([flatten_result_row(test_scores)], metrics_path)

    print(f"checkpoint_path: {checkpoint_path}")
    print_scores(test_scores)
    print(f"test_predictions_path: {predictions_path}")
    print(f"test_metrics_path: {metrics_path}")


def main() -> None:
    """Run the selected command."""
    args = parse_args()

    if args.command == "train":
        run_train_command(args)
    elif args.command == "test":
        run_test_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
