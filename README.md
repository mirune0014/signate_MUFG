# MUFG Loan Default Prediction

This repository contains a full pipeline for the SIGNATE competition on predicting loan defaults for small businesses.

## Environment
- Python 3.9+
- numpy, pandas, scikit-learn, lightgbm, xgboost, catboost, optuna, category-encoders

Install dependencies:

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost optuna category-encoders
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
3. **Train variants of LightGBM**
   - with class weights:
     ```bash
     python src/train_weighted.py
     ```
   - with target encoding features:
     ```bash
     python src/train_te.py
     ```
4. **Train alternative models**
   - XGBoost:
     ```bash
     python src/train_xgb.py
     ```
   - CatBoost:
     ```bash
     python src/train_catboost.py
     ```
5. **Search for a better decision threshold** using validation probabilities.
   ```bash
   python src/threshold_opt.py
   ```
6. **Evaluate ensemble performance** by averaging model probabilities.
   ```bash
   python src/ensemble.py
   ```
7. **(Optional) Hyperparameter optimization** for LightGBM using Optuna.
   ```bash
   python src/optimize.py
   ```
8. **Generate submission** by training final models on all data and ensembling LightGBM and XGBoost.
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
