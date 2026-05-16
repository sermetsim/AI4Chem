""" 
Utilities for preparing the glass-transition dataset. 
Remove unreliable rows, keep only the relevant columns, and split into train/validation/test sets 
that preserve the reliability distribution. 
"""


from __future__ import annotations
import numpy as np
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "PolyMetriX" / "raw" / "LAMALAB_CURATED_Tg_structured.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "PolyMetriX" / "processed"

FILTERED_DATA_PATH = PROCESSED_DATA_DIR / "tg_filtered.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "tg_train.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "tg_validation.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "tg_test.csv"

PSMILES_COLUMN = "PSMILES"
TARGET_COLUMN = "labels.Exp_Tg(K)"
RELIABILITY_COLUMN = "meta.reliability"

OUTPUT_COLUMNS = [PSMILES_COLUMN, TARGET_COLUMN, RELIABILITY_COLUMN]


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load a CSV file as a list of dictionaries."""
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def write_rows(csv_path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    """Write selected columns to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def keep_useful_tg_columns(
    rows: list[dict[str, str]],
    smiles_column: str = PSMILES_COLUMN,
    target_column: str = TARGET_COLUMN,
    reliability_column: str = RELIABILITY_COLUMN,
    reliability_to_remove: str = "red",
) -> list[dict[str, str]]:
    """Keep PSMILES, Tg, reliability, and remove unreliable rows."""
    clean_rows = []

    for row in rows:
        smiles = row[smiles_column].strip()
        target = row[target_column].strip()
        reliability = row[reliability_column].strip()

        # Remove rows with the specified reliability level (e.g., "red")
        if reliability == reliability_to_remove:
            continue
        
        # Remove rows with missing values in any of the relevant columns
        if not smiles or not target or not reliability:
            continue

        clean_rows.append(
            {
                smiles_column: smiles,
                target_column: target,
                reliability_column: reliability,
            }
        )

    return clean_rows


def split_rows_by_reliability(
    rows: list[dict[str, str]],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Split rows while preserving the reliability distribution.
    Stratified split: each reliability group is shuffled and
    split separately, then the groups are merged back together.
    """
    total_fraction = train_fraction + validation_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1.0")

    rng = random.Random(seed)
    rows_by_reliability = defaultdict(list)

    for row in rows:
        rows_by_reliability[row[RELIABILITY_COLUMN]].append(row)

    train_rows = []
    validation_rows = []
    test_rows = []

    for reliability_rows in rows_by_reliability.values():
        shuffled_rows = reliability_rows.copy()
        rng.shuffle(shuffled_rows)

        n_rows = len(shuffled_rows)
        n_train = round(n_rows * train_fraction)
        n_validation = round(n_rows * validation_fraction)

        train_rows.extend(shuffled_rows[:n_train])
        validation_rows.extend(shuffled_rows[n_train : n_train + n_validation])
        test_rows.extend(shuffled_rows[n_train + n_validation :])

    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    rng.shuffle(test_rows)

    return train_rows, validation_rows, test_rows


def print_reliability_counts(name: str, rows: list[dict[str, str]]) -> None:
    """Print how many rows of each reliability level are in a dataset."""
    counts = Counter(row[RELIABILITY_COLUMN] for row in rows)
    counts_text = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
    print(f"{name}: {len(rows)} rows ({counts_text})")


def prepare_tg_dataset() -> None:
    """Create the filtered Tg dataset and train/validation/test splits."""
    filtered_path = PROCESSED_DATA_DIR / "tg_filtered.csv"
    train_path = PROCESSED_DATA_DIR / "tg_train.csv"
    validation_path = PROCESSED_DATA_DIR / "tg_validation.csv"
    test_path = PROCESSED_DATA_DIR / "tg_test.csv"

    raw_rows = load_rows(RAW_DATA_PATH)
    filtered_rows = keep_useful_tg_columns(raw_rows)
    train_rows, validation_rows, test_rows = split_rows_by_reliability(filtered_rows)

    write_rows(filtered_path, filtered_rows, OUTPUT_COLUMNS)
    write_rows(train_path, train_rows, OUTPUT_COLUMNS)
    write_rows(validation_path, validation_rows, OUTPUT_COLUMNS)
    write_rows(test_path, test_rows, OUTPUT_COLUMNS)

    print_reliability_counts("Filtered", filtered_rows)
    print_reliability_counts("Train", train_rows)
    print_reliability_counts("Validation", validation_rows)
    print_reliability_counts("Test", test_rows)

def load_tg_csv(csv_path: Path) -> tuple[list[str], np.ndarray] :
    """Load a CSV file and return PSMILES and Tg values as numpy arrays."""
    rows = load_rows(csv_path)
    psmiles_list = []
    tg_values = []
    for row in rows:
        psmiles_list.append(row[PSMILES_COLUMN])
        tg_values.append(float(row[TARGET_COLUMN]))
    return psmiles_list, np.array(tg_values, dtype=np.float32)


if __name__ == "__main__":
    prepare_tg_dataset()
