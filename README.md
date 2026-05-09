# AI4Chem - Polymer Glass Transition Prediction

This repository contains a machine learning pipeline for predicting polymer glass transition temperature, `T_g`, from polymer SMILES-like strings (`PSMILES`).

The current goal is to build a clean baseline workflow from scratch:

- prepare the PolyMetriX `T_g` dataset;
- convert PSMILES into Morgan fingerprints with RDKit;
- train simple linear models implemented manually with NumPy;
- evaluate the models on validation and test splits;
- use notebooks to inspect data, results, and plots.

The project is intentionally educational: the first models are implemented from scratch instead of relying directly on scikit-learn.

## Current Status

Implemented:

- raw-to-processed `T_g` dataset preparation;
- stratified train/validation/test split by `meta.reliability`;
- Morgan fingerprint featurization from PSMILES;
- closed-form ordinary least squares regression;
- closed-form Ridge regression;
- regression metrics: MSE, RMSE, MAE, and R2;
- command-line entry point for baseline models;
- notebook for Ridge baseline comparison and plots.

Planned:

- Ridge regression with gradient descent;
- Lasso / Elastic Net from scratch;
- packaged baselines such as random forests or gradient boosting;
- neural models or graph-based models later.

## TODO

Short-term:

- [ ] Add Ridge regression trained with gradient descent from scratch.
- [ ] Compare closed-form Ridge and gradient-descent Ridge.
- [ ] Add validation-based alpha selection to the command-line workflow.
- [ ] Save experiment results to a small CSV or JSON file.
- [ ] Clean and complete `environment.yml`.

Modeling:

- [ ] Add Lasso regression from scratch.
- [ ] Add Elastic Net regression from scratch.
- [ ] Add packaged baselines for comparison, such as Random Forest and Gradient Boosting.
- [ ] Investigate other polymer-specific featurizers.

Analysis:

- [ ] Add parity plots and residual plots for every model.
- [ ] Analyze errors by `meta.reliability`.
- [ ] Inspect worst predictions and chemically similar polymers.
- [ ] Add a final model comparison notebook.

Engineering:

- [ ] Add proper tests for data loading, featurization, and metrics.
- [ ] Add command-line options for model choice, alpha grid, and fingerprint settings.
- [ ] Improve README results section once final baselines are stable.

## Repository Structure

```text
AI4Chem/
├── data/
│   ├── PolyMetriX/
│   │   ├── raw/
│   │   │   └── LAMALAB_CURATED_Tg_structured.csv
│   │   └── processed/
│   │       ├── tg_filtered.csv
│   │       ├── tg_train.csv
│   │       ├── tg_validation.csv
│   │       └── tg_test.csv
│   └── kaggle/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_ridge_baseline_comparison.ipynb
├── src/
│   ├── data_utils.py
│   ├── featurize.py
│   ├── models.py
│   ├── evaluate.py
│   └── train.py
├── main.py
├── environment.yml
└── README.md
```

## Data

The first dataset used is:

```text
data/PolyMetriX/raw/LAMALAB_CURATED_Tg_structured.csv
```

The preprocessing step keeps only:

- `PSMILES`
- `labels.Exp_Tg(K)`
- `meta.reliability`

Rows with reliability `red` are removed.

The processed data is split into:

- train: 70%
- validation: 15%
- test: 15%

The split is stratified by `meta.reliability`, so the reliability distribution is preserved across the three sets.

## Installation

Create and activate your environment, then install the required scientific Python packages.

Minimum packages currently needed:

- `numpy`
- `rdkit`
- `matplotlib`
- `jupyter`

Example with conda:

```bash
conda create -n ai4chem python=3.11 numpy matplotlib jupyter rdkit -c conda-forge
conda activate ai4chem
```

If the project environment file is completed later, use:

```bash
conda env create -f environment.yml
conda activate ai4chem
```

## Preparing the Data

From the repository root:

```bash
python src/data_utils.py
```

This creates:

```text
data/PolyMetriX/processed/tg_filtered.csv
data/PolyMetriX/processed/tg_train.csv
data/PolyMetriX/processed/tg_validation.csv
data/PolyMetriX/processed/tg_test.csv
```

Expected split sizes after filtering:

```text
Filtered: 7363 rows
Train: 5154 rows
Validation: 1104 rows
Test: 1105 rows
```

## Running Baseline Models

Run Ridge regression:

```bash
python main.py --model ridge --alpha 1.0
```

Run ordinary least squares:

```bash
python main.py --model linear
```

Note: closed-form OLS may fail with a singular matrix on Morgan fingerprints. This is expected because fingerprint columns can be redundant or linearly dependent. Ridge regression is the preferred baseline for this feature representation.

## Notebooks

Use the notebooks for exploration and plots:

```text
notebooks/01_data_exploration.ipynb
notebooks/02_ridge_baseline_comparison.ipynb
```

The Ridge comparison notebook:

- compares train/validation/test target distributions;
- checks reliability distributions;
- featurizes PSMILES;
- confirms whether OLS fails;
- sweeps Ridge `alpha`;
- plots RMSE, MAE, and R2;
- plots true vs predicted `T_g`;
- plots residual distributions.

## Code Overview

### `src/data_utils.py`

Handles dataset preparation and CSV loading:

- load raw CSV rows;
- keep useful columns;
- remove unreliable rows;
- split train/validation/test;
- load processed CSVs into PSMILES lists and target arrays.

### `src/featurize.py`

Handles chemical featurization:

- convert PSMILES to RDKit molecules;
- generate Morgan fingerprints;
- convert fingerprints into NumPy arrays.

Feature matrix convention:

```text
X.shape = (n_molecules, n_features)
```

For Morgan fingerprints:

```text
n_features = 2048
```

### `src/models.py`

Contains models implemented from scratch:

- `LinearRegressionClosedForm`
- `RidgeRegressionClosedForm`

Both use the interface:

```python
model.fit(X_train, y_train)
y_pred = model.predict(X)
```

### `src/evaluate.py`

Contains regression metrics:

- MSE
- RMSE
- MAE
- R2

### `src/train.py`

Connects the full baseline workflow:

```text
load data -> featurize -> fit model -> predict -> evaluate
```

### `main.py`

Command-line entry point.

Example:

```bash
python main.py --model ridge --alpha 10.0
```

## First Baseline Interpretation

The Ridge baseline with Morgan fingerprints gives a first estimate of how much information the fingerprint representation contains for predicting `T_g`.

MSE is reported in squared Kelvin, so RMSE and MAE are usually easier to interpret:

```text
RMSE: typical error scale in K
MAE: average absolute error in K
R2: fraction of target variance explained
```

## Development Notes

Recommended workflow when adding a new model:

1. Add the model class or wrapper in `src/models.py`.
2. Reuse `load_and_featurize_data` from `src/train.py`.
3. Reuse metrics from `src/evaluate.py`.
4. Add a training function in `src/train.py`.
5. Add a command-line option in `main.py`.
6. Compare results in a notebook.

Keep responsibilities separated:

```text
data_utils.py -> data loading and splitting
featurize.py  -> PSMILES to numerical features
models.py     -> model definitions
evaluate.py   -> metrics
train.py      -> training workflows
main.py       -> command-line interface
```

## License

See [LICENSE](LICENSE).
