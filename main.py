"""Command-line entry point for running AI4Chem experiments."""

import argparse

from src.train import run_linear_regression_experiment, run_ridge_regression_experiment


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run AI4Chem baseline experiments.")
    parser.add_argument(
        "--model",
        choices=["linear", "ridge"],
        default="linear",
        help="Model to run.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength. Used only when --model ridge.",
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
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print_scores(scores)


if __name__ == "__main__":
    main()
