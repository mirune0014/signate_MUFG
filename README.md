# MUFG Loan Default Prediction

This repository contains a full pipeline for the SIGNATE competition on predicting loan defaults for small businesses.

## Environment
- Python 3.9+
- numpy, pandas, scikit-learn, lightgbm, xgboost, optuna

Install dependencies:

```bash
pip install numpy pandas scikit-learn lightgbm xgboost optuna
```

## Usage
1. **Preprocess** the raw CSV files and generate feature matrices.
   ```bash
   python src/preprocess.py
   ```
2. **Train LightGBM baseline** and record cross-validation scores.
   ```bash
   python src/train.py
   ```
3. **Train XGBoost model** for comparison.
   ```bash
   python src/train_xgb.py
   ```
4. **(Optional) Hyperparameter optimization** for LightGBM using Optuna.
   ```bash
   python src/optimize.py
   ```
5. **Generate submission** by training final models on all data and ensembling LightGBM and XGBoost.
   ```bash
   python src/predict.py
   ```
   The file `data/output/submit.csv` will be created in the sample submission format.

## Repository Structure
- `data/input/` – raw competition data
- `data/output/` – processed features, evaluation results, and submission file
- `src/` – preprocessing, training, optimization, and inference scripts
- `tasks.md` – checklist of steps toward a complete submission
- `explain.md` – competition overview and additional documentation
