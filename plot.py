"""Create report figures from saved experiment CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_utils import TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH, load_tg_csv


RESULTS_DIR = Path("artifacts") / "results"
FIGURES_DIR = Path("report") / "figures"
COLORMAP_NAME = "plasma"


def get_palette(n_colors: int) -> list:
    """Return colors sampled from the project colormap."""
    cmap = plt.colormaps[COLORMAP_NAME]
    if n_colors <= 1:
        return [cmap(0.55)]
    return [cmap(value) for value in np.linspace(0.15, 0.85, n_colors)]


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV file as a list of dictionaries."""
    with csv_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_float(value: str) -> float:
    """Convert a CSV value to float."""
    return float(value)


def parse_json_dict(value: str) -> dict:
    """Parse a JSON dictionary stored inside a CSV cell."""
    if not value:
        return {}
    return json.loads(value)


def discover_result_files(results_dir: Path) -> dict[str, dict[str, Path]]:
    """Find summary, validation, metric, and prediction CSV files by model name."""
    result_files: dict[str, dict[str, Path]] = {}

    for summary_path in results_dir.rglob("*_summary.csv"):
        rows = read_csv_rows(summary_path)
        if not rows:
            continue
        model_name = rows[0]["model_name"]
        result_files.setdefault(model_name, {})["summary"] = summary_path

    for validation_path in results_dir.rglob("*_validation_sweep.csv"):
        rows = read_csv_rows(validation_path)
        if not rows:
            continue
        model_name = rows[0]["model_name"]
        result_files.setdefault(model_name, {})["validation"] = validation_path

    for metrics_path in results_dir.rglob("*_test_metrics.csv"):
        model_name = metrics_path.name.removesuffix("_test_metrics.csv")
        result_files.setdefault(model_name, {})["test_metrics"] = metrics_path

    for predictions_path in results_dir.rglob("*_test_predictions.csv"):
        model_name = predictions_path.name.removesuffix("_test_predictions.csv")
        result_files.setdefault(model_name, {})["test_predictions"] = predictions_path

    return result_files


def load_summary_table(result_files: dict[str, dict[str, Path]]) -> list[dict]:
    """Load one summary row per model."""
    rows = []
    for model_name, paths in sorted(result_files.items()):
        if "summary" not in paths:
            continue

        summary = read_csv_rows(paths["summary"])[0]
        rows.append(
            {
                "model_name": model_name,
                "best_params": parse_json_dict(summary.get("best_params", "{}")),
                "best_validation_rmse": as_float(summary["best_validation_rmse"]),
                "rmse_test": as_float(summary["rmse_test"]),
                "mae_test": as_float(summary["mae_test"]),
                "mse_test": as_float(summary["mse_test"]),
                "r2_test": as_float(summary["r2_test"]),
            }
        )
    return rows


def load_test_metrics_table(result_files: dict[str, dict[str, Path]]) -> list[dict]:
    """Load one explicit test-metrics row per model."""
    rows = []
    for model_name, paths in sorted(result_files.items()):
        if "test_metrics" not in paths:
            continue

        metrics = read_csv_rows(paths["test_metrics"])[0]
        rows.append(
            {
                "model_name": model_name,
                "rmse_test": as_float(metrics["rmse_test"]),
                "mae_test": as_float(metrics["mae_test"]),
                "mse_test": as_float(metrics["mse_test"]),
                "r2_test": as_float(metrics["r2_test"]),
            }
        )
    return rows


def load_validation_results(validation_path: Path) -> list[dict]:
    """Load validation sweep rows."""
    rows = []
    for row in read_csv_rows(validation_path):
        rows.append(
            {
                "model_name": row["model_name"],
                "params": parse_json_dict(row["params"]),
                "rmse_validation": as_float(row["rmse_validation"]),
            }
        )
    return rows


def load_test_predictions(predictions_path: Path) -> dict[str, np.ndarray]:
    """Load y_true and y_pred arrays from a test prediction CSV."""
    rows = read_csv_rows(predictions_path)
    return {
        "y_true": np.array([as_float(row["y_true"]) for row in rows]),
        "y_pred": np.array([as_float(row["y_pred"]) for row in rows]),
    }


def short_params_label(params: dict) -> str:
    """Create a compact label from a parameter dictionary."""
    return ", ".join(f"{key}={value}" for key, value in params.items())


def get_single_numeric_param(results: list[dict]) -> tuple[str, np.ndarray] | None:
    """Return a numeric hyperparameter axis if every result varies one same numeric key."""
    if not results:
        return None

    keys = set(results[0]["params"])
    if len(keys) != 1:
        return None

    key = next(iter(keys))
    values = []
    for result in results:
        params = result["params"]
        if set(params) != {key}:
            return None
        value = params[key]
        if not isinstance(value, (int, float)):
            return None
        values.append(float(value))

    return key, np.array(values)


def thin_points_by_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_bins: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep one point per 2D grid cell to reduce overplotting."""
    if len(x_values) == 0:
        return x_values, y_values

    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()

    if x_min == x_max or y_min == y_max:
        return x_values, y_values

    x_bins = np.linspace(x_min, x_max, n_bins + 1)
    y_bins = np.linspace(y_min, y_max, n_bins + 1)
    x_indices = np.digitize(x_values, x_bins) - 1
    y_indices = np.digitize(y_values, y_bins) - 1

    seen_cells = set()
    keep_indices = []

    for index, cell in enumerate(zip(x_indices, y_indices)):
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
        keep_indices.append(index)

    keep_indices = np.array(keep_indices)
    return x_values[keep_indices], y_values[keep_indices]


def save_current_figure(output_path: Path) -> None:
    """Save the current matplotlib figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tg_distribution(figures_dir: Path) -> None:
    """Plot Tg distributions for train, validation, and test splits."""
    _, y_train = load_tg_csv(TRAIN_DATA_PATH)
    _, y_validation = load_tg_csv(VALIDATION_DATA_PATH)
    _, y_test = load_tg_csv(TEST_DATA_PATH)
    colors = get_palette(3)

    plt.figure(figsize=(7, 4))
    plt.hist(y_train, bins=40, alpha=0.55, label="train", color=colors[0])
    plt.hist(y_validation, bins=40, alpha=0.55, label="validation", color=colors[1])
    plt.hist(y_test, bins=40, alpha=0.55, label="test", color=colors[2])
    plt.xlabel("Experimental $T_g$ (K)")
    plt.ylabel("Count")
    plt.legend()
    save_current_figure(figures_dir / "tg_distribution.png")


def plot_validation_rmse(result_files: dict[str, dict[str, Path]], figures_dir: Path) -> None:
    """Plot validation RMSE for hyperparameter tuning."""
    validation_items = [
        (model_name, paths["validation"])
        for model_name, paths in sorted(result_files.items())
        if "validation" in paths
    ]
    if not validation_items:
        print("No validation sweep CSV files found.")
        return

    n_models = len(validation_items)
    validation_lengths = [
        len(load_validation_results(validation_path))
        for _, validation_path in validation_items
    ]
    figure_height = sum(max(3.5, 0.35 * length) for length in validation_lengths)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, figure_height))
    if n_models == 1:
        axes = [axes]
    colors = get_palette(n_models)

    for ax, color, (model_name, validation_path) in zip(axes, colors, validation_items):
        results = load_validation_results(validation_path)
        rmse = np.array([row["rmse_validation"] for row in results])
        numeric_axis = get_single_numeric_param(results)

        if numeric_axis is not None:
            param_name, x_values = numeric_axis
            order = np.argsort(x_values)
            ax.plot(x_values[order], rmse[order], marker="o", color=color)
            ax.set_xlabel(param_name)
            if np.all(x_values > 0) and x_values.max() / x_values.min() >= 10:
                ax.set_xscale("log")
        else:
            x_values = np.arange(1, len(results) + 1)
            ax.plot(x_values, rmse, marker="o", color=color)
            ax.set_xticks(x_values)
            ax.set_xticklabels([str(value) for value in x_values])
            ax.set_xlabel("hyperparameter setting")

            setting_text = "\n".join(
                f"{index}: {short_params_label(row['params'])}"
                for index, row in zip(x_values, results)
            )
            ax.text(
                1.02,
                0.5,
                setting_text,
                transform=ax.transAxes,
                va="center",
                fontsize=8,
            )

        ax.set_ylabel("Validation RMSE (K)")
        ax.grid(alpha=0.3)

    save_current_figure(figures_dir / "validation_rmse_hyperparameter_tuning.png")

    per_model_dir = figures_dir / "validation_rmse_by_model"
    for model_name, validation_path in validation_items:
        results = load_validation_results(validation_path)
        rmse = np.array([row["rmse_validation"] for row in results])
        numeric_axis = get_single_numeric_param(results)
        color = get_palette(1)[0]

        plt.figure(figsize=(7, 4))
        if numeric_axis is not None:
            param_name, x_values = numeric_axis
            order = np.argsort(x_values)
            plt.plot(x_values[order], rmse[order], marker="o", color=color)
            plt.xlabel(param_name)
            if np.all(x_values > 0) and x_values.max() / x_values.min() >= 10:
                plt.xscale("log")
        else:
            x_values = np.arange(1, len(results) + 1)
            plt.plot(x_values, rmse, marker="o", color=color)
            plt.xticks(x_values, [str(value) for value in x_values])
            plt.xlabel("hyperparameter setting")
            setting_text = "\n".join(
                f"{index}: {short_params_label(row['params'])}"
                for index, row in zip(x_values, results)
            )
            plt.gca().text(
                1.02,
                0.5,
                setting_text,
                transform=plt.gca().transAxes,
                va="center",
                fontsize=8,
            )

        plt.ylabel("Validation RMSE (K)")
        plt.grid(alpha=0.3)
        save_current_figure(per_model_dir / f"{model_name}_validation_rmse.png")


def plot_metric_comparison(
    test_metric_rows: list[dict],
    summary_rows: list[dict],
    metric_key: str,
    ylabel: str,
    output_path: Path,
    summary_marker_key: str | None = None,
    summary_marker_label: str | None = None,
) -> None:
    """Plot a bar chart comparing one final test metric across models."""
    if not test_metric_rows:
        print(f"No test metrics CSV files found for {metric_key}.")
        return

    summary_by_model = {row["model_name"]: row for row in summary_rows}
    model_names = [row["model_name"] for row in test_metric_rows]
    values = [row[metric_key] for row in test_metric_rows]
    colors = get_palette(len(model_names))

    plt.figure(figsize=(7, 4))
    x_values = np.arange(len(model_names))
    bars = plt.bar(x_values, values, color=colors)

    for x_value, model_name, value in zip(x_values, model_names, values):
        summary_value = None
        if summary_marker_key is not None:
            summary_value = summary_by_model.get(model_name, {}).get(summary_marker_key)

        plt.text(
            x_value,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        if summary_value is not None:
            plt.hlines(
                y=summary_value,
                xmin=x_value - 0.35,
                xmax=x_value + 0.35,
                colors="black",
                linestyles="dotted",
                linewidth=2.0,
            )

    plt.xticks(x_values, model_names)
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.grid(axis="y", alpha=0.3)
    if summary_marker_key is not None:
        label = summary_marker_label or summary_marker_key
        plt.plot([], [], color="black", linestyle="dotted", label=label)
        plt.legend(loc="lower left")
    save_current_figure(output_path)


def plot_parity(
    result_files: dict[str, dict[str, Path]],
    figures_dir: Path,
) -> None:
    """Plot predicted vs experimental Tg for each model."""
    prediction_items = [
        (model_name, paths["test_predictions"])
        for model_name, paths in sorted(result_files.items())
        if "test_predictions" in paths
    ]
    if not prediction_items:
        print("No test prediction CSV files found for parity plot.")
        return

    n_models = len(prediction_items)
    n_cols = min(2, n_models)
    n_rows = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)
    colors = get_palette(n_models)

    for ax, color, (model_name, predictions_path) in zip(axes, colors, prediction_items):
        predictions = load_test_predictions(predictions_path)
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]

        ax.scatter(y_true, y_pred, s=14, alpha=0.55, color=color)
        low = min(y_true.min(), y_pred.min())
        high = max(y_true.max(), y_pred.max())
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set_xlabel("Experimental $T_g$ (K)")
        ax.set_ylabel("Predicted $T_g$ (K)")
        ax.grid(alpha=0.3)

    for ax in axes[len(prediction_items):]:
        ax.axis("off")

    save_current_figure(figures_dir / "parity_predicted_vs_experimental_tg.png")

    per_model_dir = figures_dir / "parity_by_model"
    for model_name, predictions_path in prediction_items:
        predictions = load_test_predictions(predictions_path)
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]
        color = get_palette(1)[0]

        plt.figure(figsize=(5, 4.5))
        plt.scatter(y_true, y_pred, s=14, alpha=0.55, color=color)
        low = min(y_true.min(), y_pred.min())
        high = max(y_true.max(), y_pred.max())
        plt.plot([low, high], [low, high], color="black", linewidth=1)
        plt.xlabel("Experimental $T_g$ (K)")
        plt.ylabel("Predicted $T_g$ (K)")
        plt.grid(alpha=0.3)
        save_current_figure(per_model_dir / f"{model_name}_parity.png")


def plot_parity_overlay(
    result_files: dict[str, dict[str, Path]],
    figures_dir: Path,
) -> None:
    """Plot predicted vs experimental Tg for all models on one axis."""
    prediction_items = [
        (model_name, paths["test_predictions"])
        for model_name, paths in sorted(result_files.items())
        if "test_predictions" in paths
    ]
    if not prediction_items:
        print("No test prediction CSV files found for parity overlay plot.")
        return

    colors = get_palette(len(prediction_items))
    global_low = float("inf")
    global_high = float("-inf")

    plt.figure(figsize=(6, 5))
    for color, (model_name, predictions_path) in zip(colors, prediction_items):
        predictions = load_test_predictions(predictions_path)
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]
        global_low = min(global_low, y_true.min(), y_pred.min())
        global_high = max(global_high, y_true.max(), y_pred.max())
        y_true_thin, y_pred_thin = thin_points_by_grid(y_true, y_pred)
        plt.scatter(
            y_true_thin,
            y_pred_thin,
            s=12,
            alpha=0.4,
            color=color,
            label=f"{model_name} (n={len(y_true_thin)})",
        )

    plt.plot([global_low, global_high], [global_low, global_high], color="black", linewidth=1)
    plt.xlabel("Experimental $T_g$ (K)")
    plt.ylabel("Predicted $T_g$ (K)")
    plt.legend(markerscale=2)
    plt.grid(alpha=0.3)
    save_current_figure(figures_dir / "parity_predicted_vs_experimental_tg_overlay.png")


def plot_residual_distributions(
    result_files: dict[str, dict[str, Path]],
    figures_dir: Path,
) -> None:
    """Plot residual density distributions for each model."""
    prediction_items = [
        (model_name, paths["test_predictions"])
        for model_name, paths in sorted(result_files.items())
        if "test_predictions" in paths
    ]
    if not prediction_items:
        print("No test prediction CSV files found for residual plot.")
        return

    residuals_by_model = []
    for model_name, predictions_path in prediction_items:
        predictions = load_test_predictions(predictions_path)
        residuals = predictions["y_true"] - predictions["y_pred"]
        residuals_by_model.append((model_name, residuals))

    all_residuals = np.concatenate([residuals for _, residuals in residuals_by_model])
    common_bins = np.linspace(all_residuals.min(), all_residuals.max(), 45)

    n_models = len(residuals_by_model)
    n_cols = min(2, n_models)
    n_rows = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    colors = get_palette(n_models)

    for ax, color, (model_name, residuals) in zip(axes, colors, residuals_by_model):
        ax.hist(residuals, bins=common_bins, density=True, alpha=0.75, color=color)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Residual $T_g$ true - predicted (K)")
        ax.set_ylabel("Density")

    for ax in axes[len(residuals_by_model):]:
        ax.axis("off")

    save_current_figure(figures_dir / "residual_distributions.png")

    per_model_dir = figures_dir / "residual_distributions_by_model"
    color = get_palette(1)[0]
    for model_name, residuals in residuals_by_model:
        plt.figure(figsize=(5.5, 4))
        plt.hist(residuals, bins=common_bins, density=True, alpha=0.75, color=color)
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("Residual $T_g$ true - predicted (K)")
        plt.ylabel("Density")
        save_current_figure(per_model_dir / f"{model_name}_residual_distribution.png")


def plot_residuals_vs_experimental_overlay(
    result_files: dict[str, dict[str, Path]],
    figures_dir: Path,
) -> None:
    """Plot residuals versus experimental Tg for all models on one axis."""
    prediction_items = [
        (model_name, paths["test_predictions"])
        for model_name, paths in sorted(result_files.items())
        if "test_predictions" in paths
    ]
    if not prediction_items:
        print("No test prediction CSV files found for residual overlay plot.")
        return

    colors = get_palette(len(prediction_items))

    plt.figure(figsize=(6, 5))
    for color, (model_name, predictions_path) in zip(colors, prediction_items):
        predictions = load_test_predictions(predictions_path)
        y_true = predictions["y_true"]
        residuals = y_true - predictions["y_pred"]
        y_true_thin, residuals_thin = thin_points_by_grid(y_true, residuals)
        plt.scatter(
            y_true_thin,
            residuals_thin,
            s=12,
            alpha=0.4,
            color=color,
            label=f"{model_name} (n={len(y_true_thin)})",
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Experimental $T_g$ (K)")
    plt.ylabel("Residual true - predicted (K)")
    plt.legend(markerscale=2)
    plt.grid(alpha=0.3)
    save_current_figure(figures_dir / "residuals_vs_experimental_tg_overlay.png")


def plot_absolute_error_boxplot(
    result_files: dict[str, dict[str, Path]],
    figures_dir: Path,
) -> None:
    """Plot absolute test error distribution for each model."""
    model_names = []
    absolute_errors = []

    for model_name, paths in sorted(result_files.items()):
        if "test_predictions" not in paths:
            continue

        predictions = load_test_predictions(paths["test_predictions"])
        errors = np.abs(predictions["y_true"] - predictions["y_pred"])
        model_names.append(model_name)
        absolute_errors.append(errors)

    if not absolute_errors:
        print("No test prediction CSV files found for absolute error boxplot.")
        return

    plt.figure(figsize=(7, 4))
    box = plt.boxplot(absolute_errors, tick_labels=model_names, showfliers=True, patch_artist=True)
    colors = get_palette(len(model_names))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    plt.ylabel("Absolute error (K)")
    plt.xlabel("Model")
    plt.grid(axis="y", alpha=0.3)
    save_current_figure(figures_dir / "absolute_error_boxplot_per_model.png")


def create_all_plots(results_dir: Path, figures_dir: Path) -> None:
    """Create all report figures from saved result CSV files."""
    result_files = discover_result_files(results_dir)
    summary_rows = load_summary_table(result_files)
    test_metric_rows = load_test_metrics_table(result_files)

    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_tg_distribution(figures_dir)
    plot_validation_rmse(result_files, figures_dir)
    plot_metric_comparison(
        test_metric_rows,
        summary_rows,
        metric_key="rmse_test",
        ylabel="Final Test RMSE (K)",
        output_path=figures_dir / "final_test_rmse_comparison.png",
        summary_marker_key="best_validation_rmse",
        summary_marker_label="best validation RMSE",
    )
    plot_metric_comparison(
        test_metric_rows,
        summary_rows,
        metric_key="mae_test",
        ylabel="Final Test MAE (K)",
        output_path=figures_dir / "final_test_mae_comparison.png",
    )
    plot_parity(result_files, figures_dir)
    plot_parity_overlay(result_files, figures_dir)
    plot_residual_distributions(result_files, figures_dir)
    plot_residuals_vs_experimental_overlay(result_files, figures_dir)
    plot_absolute_error_boxplot(result_files, figures_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create report figures from result CSV files.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing saved experiment result CSV files.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory where figures will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run plotting script."""
    args = parse_args()
    create_all_plots(args.results_dir, args.figures_dir)
    print(f"Figures saved in: {args.figures_dir}")


if __name__ == "__main__":
    main()
