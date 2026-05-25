from pathlib import Path
import csv
import json
import joblib


def save_model(model: object, model_path: Path) -> None:
    """Save a trained model to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path) -> object:
    """Load a trained model from disk."""
    return joblib.load(model_path)


def save_dicts_to_csv(rows: list[dict], csv_path: Path) -> None:
    """Save a list of dictionaries to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("Cannot save an empty list of rows to CSV.")

    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result_row(row: dict) -> dict:
    """Flatten nested dictionaries so they can be written cleanly to CSV."""
    flat_row = {}

    for key, value in row.items():
        if isinstance(value, dict):
            flat_row[key] = json.dumps(value, sort_keys=True)
        else:
            flat_row[key] = value

    return flat_row
