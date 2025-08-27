import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def add_features(df: pd.DataFrame,
                 train: pd.DataFrame,
                 target_col: str) -> pd.DataFrame:
    df = df.copy()
    interest_mean = train['InitialInterestRate'].mean()
    interest_bins = np.quantile(train['InitialInterestRate'], [0, 0.25, 0.5, 0.75, 1.0])
    business_age_map = {
        'Startup, Loan Funds will Open Business': 0,
        'New Business or 2 years or less': 2,
        'Existing or more than 2 years old': 5,
        'Change of Ownership': 5,
        'Unanswered': -1,
    }
    naics_interest_mean = train.groupby('NaicsSector')['InitialInterestRate'].mean()
    naics_gross_mean = train.groupby('NaicsSector')['GrossApproval'].mean()
    program_default = train.groupby('Subprogram')[target_col].mean()
    sector_default = train.groupby('NaicsSector')[target_col].mean()
    program_gross_mean = train.groupby('Subprogram')['GrossApproval'].mean()
    sector_term_mean = train.groupby('NaicsSector')['TermInMonths'].mean()
    program_term_mean = train.groupby('Subprogram')['TermInMonths'].mean()
    overall_default = train[target_col].mean()
    gross_bins = pd.qcut(train['GrossApproval'], q=5, retbins=True, duplicates='drop')[1]
    term_bins = pd.qcut(train['TermInMonths'], q=5, retbins=True, duplicates='drop')[1]

    df['GrossApproval_log'] = np.log1p(df['GrossApproval'])
    df['SBAGuaranteedApproval_log'] = np.log1p(df['SBAGuaranteedApproval'])
    df['MonthlyPayment'] = df['GrossApproval'] / df['TermInMonths']
    df['ApprovalRatio'] = df['SBAGuaranteedApproval'] / df['GrossApproval']
    df['ApprovalDiff'] = df['GrossApproval'] - df['SBAGuaranteedApproval']
    df['InterestRateDiff'] = df['InitialInterestRate'] - interest_mean
    df['InterestRateBucket'] = np.digitize(
        df['InitialInterestRate'], interest_bins[1:-1], right=True
    ).astype(str)
    df['BusinessAgeNum'] = df['BusinessAge'].map(business_age_map)
    df['InterestRatePerTerm'] = df['InitialInterestRate'] / df['TermInMonths']
    df['JobsSupportedRatio'] = df['JobsSupported'] / df['GrossApproval']
    df['TermBucket'] = pd.cut(
        df['TermInMonths'], bins=[0, 60, 120, 180, 240, 360], labels=False, include_lowest=True
    ).astype(str)
    df['ProgramBusinessType'] = df['Subprogram'] + '_' + df['BusinessType']
    df['SectorInterestDiff'] = df['InitialInterestRate'] - df['NaicsSector'].map(naics_interest_mean)
    df['SectorApprovalDiff'] = df['GrossApproval'] - df['NaicsSector'].map(naics_gross_mean)
    df['ProgramDefaultDiff'] = df['Subprogram'].map(program_default) - overall_default
    df['SectorDefaultDiff'] = df['NaicsSector'].map(sector_default) - overall_default
    df['ProgramLoanDiff'] = df['GrossApproval'] - df['Subprogram'].map(program_gross_mean)
    df['SectorTermDiff'] = df['TermInMonths'] - df['NaicsSector'].map(sector_term_mean)
    df['ProgramTermDiff'] = df['TermInMonths'] - df['Subprogram'].map(program_term_mean)
    df['SectorProgram'] = df['NaicsSector'] + '_' + df['Subprogram']
    df['GrossApprovalBin'] = pd.cut(
        df['GrossApproval'], bins=gross_bins, labels=False, include_lowest=True
    ).astype(str)
    df['TermInMonthsBin'] = pd.cut(
        df['TermInMonths'], bins=term_bins, labels=False, include_lowest=True
    ).astype(str)
    return df


def main(trials: int) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "input"
    output_dir = base_dir / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / 'train.csv')
    target_col = 'LoanStatus'
    id_col = 'id'

    X = add_features(train.drop(columns=[target_col]), train, target_col)
    y = train[target_col].values

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    X = X.drop(columns=[id_col])

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "loss_function": "Logloss",
            "eval_metric": "F1",
            "verbose": False,
            "random_state": 42,
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, valid_idx in cv.split(X, y):
            model = CatBoostClassifier(**params)
            model.fit(X.iloc[train_idx], y[train_idx], cat_features=cat_cols)
            preds = (model.predict_proba(X.iloc[valid_idx])[:, 1] > 0.5).astype(int)
            scores.append(f1_score(y[valid_idx], preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)

    best = {"best_params": study.best_params, "best_score": study.best_value}
    with open(output_dir / "cat_optuna_results.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    main(args.trials)
