# AI4Chem - Glass Temperature Assessment (Glass-TA)

This repository contains a machine learning workflow for predicting polymer glass transition temperature, \(T_g\), from polymer PSMILES strings.

The pipeline starts from the PolyMetriX raw `T_g` dataset, creates clean train/validation/test splits, converts PSMILES to Morgan fingerprints with RDKit, trains several regression models, saves model checkpoints and result CSV files, and generates report figures.

## Implemented Workflow

- Raw PolyMetriX data preparation.
- Stratified train/validation/test split by `meta.reliability`.
- RDKit Morgan fingerprint featurization from PSMILES.
- Closed-form linear regression and Ridge regression implemented with NumPy.
- Packaged model wrappers for:
  - Random Forest
  - XGBoost
  - Support Vector Regression
  - Elastic Net
- Hyperparameter selection using validation RMSE.
- Final retraining on train + validation.
- Final evaluation on test data.
- Model checkpoint saving with `joblib`.
- CSV export for validation sweeps, final summaries, test metrics, and test predictions.
- Report figure generation from saved CSV outputs.

## Repository Structure

```text
AI4Chem/
├── artifacts/
│   ├── models/
│   │   └── best_<model>.joblib
│   └── results/
│       └── <model>/
│           ├── <model>_summary.csv
│           ├── <model>_validation_sweep.csv
│           ├── <model>_test_metrics.csv
│           └── <model>_test_predictions.csv
├── data/
│   └── PolyMetriX/
│       ├── raw/
│       │   └── LAMALAB_CURATED_Tg_structured.csv
│       └── processed/
│           ├── tg_filtered.csv
│           ├── tg_train.csv
│           ├── tg_validation.csv
│           └── tg_test.csv
├── report/
│   └── figures/
├── src/
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── featurize.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── main.py
├── plot.py
├── environment.yml
└── README.md
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate ai4chem
```

## Data Preparation

Prepare the PolyMetriX `T_g` dataset:

```bash
python src/data_utils.py
```

This keeps:

- `PSMILES`
- `labels.Exp_Tg(K)`
- `meta.reliability`

Rows with reliability `red` are removed.

The processed files are:

```text
data/PolyMetriX/processed/tg_filtered.csv
data/PolyMetriX/processed/tg_train.csv
data/PolyMetriX/processed/tg_validation.csv
data/PolyMetriX/processed/tg_test.csv
```

Expected split sizes:

```text
Filtered: 7363 rows
Train: 5154 rows
Validation: 1104 rows
Test: 1105 rows
```

## Training

Train a model with hyperparameter selection:

```bash
python main.py train --model ridge
```

Available models:

```text
ridge
randomforest
xgboost
svm
elasticnet
```

Training does the following:

1. Loads and featurizes train, validation, and test splits.
2. Trains every hyperparameter configuration on the train split.
3. Selects the best configuration using validation RMSE.
4. Retrains the best model on train + validation.
5. Evaluates the final model on the test split.
6. Saves model and result artifacts.

Default outputs for `ridge`:

```text
artifacts/models/best_ridge.joblib
artifacts/results/ridge/ridge_summary.csv
artifacts/results/ridge/ridge_validation_sweep.csv
```

Custom output paths:

```bash
python main.py train --model ridge \
  --output artifacts/models/my_ridge.joblib \
  --results-output artifacts/results/ridge/my_ridge_summary.csv \
  --validation-output artifacts/results/ridge/my_ridge_validation_sweep.csv
```

## Testing A Saved Model

Evaluate a saved checkpoint on the test split:

```bash
python main.py test --model ridge
```

Default outputs:

```text
artifacts/results/ridge/ridge_test_metrics.csv
artifacts/results/ridge/ridge_test_predictions.csv
```

Custom checkpoint and output paths:

```bash
python main.py test --model ridge \
  --checkpoint artifacts/models/best_ridge.joblib \
  --metrics-output artifacts/results/ridge/custom_test_metrics.csv \
  --predictions-output artifacts/results/ridge/custom_test_predictions.csv
```

## Plotting

Generate report figures from saved CSV artifacts:

```bash
python plot.py
```

Figures are saved in:

```text
report/figures/
```

Generated figures include:

- \(T_g\) distribution over train/validation/test splits.
- Validation RMSE hyperparameter sweeps.
- Final test RMSE comparison.
- Final test MAE comparison.
- Predicted vs experimental \(T_g\) parity plots.
- Residual distribution plots.
- Residual-vs-experimental overlay plots.
- Absolute error boxplot per model.

## Code Overview

### `src/data_utils.py`

Dataset preparation and CSV loading:

- load raw CSV rows;
- keep relevant `T_g` columns;
- remove unreliable rows;
- create stratified train/validation/test splits;
- load processed CSVs into PSMILES lists and target arrays.

### `src/featurize.py`

Chemical featurization:

- PSMILES to RDKit molecule;
- molecule to Morgan fingerprint;
- Morgan fingerprint to NumPy feature matrix.

Feature convention:

```text
X.shape = (n_molecules, n_features)
```

The current Morgan fingerprint size is 2048 bits.

### `src/models.py`

Model definitions and factory:

- `LinearRegressionClosedForm`
- `RidgeRegressionClosedForm`
- `RandomForest`
- `XGBoostModel`
- `SupportVectorModel`
- `ElasticNetModel`
- `create_model`

All models expose:

```python
model.fit(X_train, y_train)
y_pred = model.predict(X)
```

### `src/train.py`

Training workflow:

- default hyperparameter grids;
- train/validation/test loading and featurization;
- hyperparameter selection;
- final retraining;
- final test evaluation;
- checkpoint saving.

### `src/evaluate.py`

Regression metrics:

- MSE
- RMSE
- MAE
- \(R^2\)

### `src/utils.py`

General utilities:

- save/load model checkpoints;
- write dictionaries to CSV;
- flatten nested result dictionaries for CSV output.

### `main.py`

Command-line interface for training and testing.

### `plot.py`

Figure generation from saved result CSV files.

## Notes

Closed-form ordinary least squares can fail with a singular matrix on Morgan fingerprints because fingerprint features may be redundant or linearly dependent. Ridge regression is more stable and is the main linear baseline.

The `train` command already evaluates the final retrained model on the test set. The `test` command is useful for reloading a saved checkpoint and regenerating test predictions/metrics without retraining.

Concerning the .joblib files, SVM and random forest models cannot be saved on GitHub because their size are too large (larger than 100 Mo). Nevertheless, the best parameters are available in artifacts/results.

## License

See [LICENSE](LICENSE).
