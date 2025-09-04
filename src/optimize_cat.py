import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def add_features(df: pd.DataFrame,
                 interest_mean: float,
                 interest_bins: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    business_age_map = {
        'Startup, Loan Funds will Open Business': 0,
        'New Business or 2 years or less': 2,
        'Existing or more than 2 years old': 5,
        'Change of Ownership': 5,
        'Unanswered': -1,
    }
    df['GrossApproval_log'] = np.log1p(df['GrossApproval'])
    df['SBAGuaranteedApproval_log'] = np.log1p(df['SBAGuaranteedApproval'])
    df['TermInYears'] = df['TermInMonths'] / 12
    df['MonthlyPayment'] = df['GrossApproval'] / df['TermInMonths']
    df['ApprovalRatio'] = df['SBAGuaranteedApproval'] / df['GrossApproval']
    df['ApprovalDiff'] = df['GrossApproval'] - df['SBAGuaranteedApproval']
    df['GuaranteeGapRatio'] = df['ApprovalDiff'] / df['GrossApproval']
    df['InterestRateDiff'] = df['InitialInterestRate'] - interest_mean
    df['InterestRateBucket'] = np.digitize(
        df['InitialInterestRate'], interest_bins[1:-1], right=True
    ).astype(str)
    df['BusinessAgeNum'] = df['BusinessAge'].map(business_age_map)
    df['InterestRatePerTerm'] = df['InitialInterestRate'] / df['TermInMonths']
    df['JobsSupportedRatio'] = df['JobsSupported'] / df['GrossApproval']
    df['Jobs_log'] = np.log1p(df['JobsSupported'])
    df['GrossPerJob'] = df['GrossApproval'] / df['JobsSupported'].replace(0, np.nan)
    df['GuaranteePerJob'] = df['SBAGuaranteedApproval'] / df['JobsSupported'].replace(0, np.nan)
    df['TermBucket'] = pd.cut(
        df['TermInMonths'], bins=[0, 60, 120, 180, 240, 360], labels=False, include_lowest=True
    ).astype(str)
    df['ProgramBusinessType'] = df['Subprogram'] + '_' + df['BusinessType']
    df['RateTerm'] = df['InterestRateBucket'] + '_' + df['TermBucket']
    df['HasCollateral'] = (df['CollateralInd'] == 'Y').astype(int)
    df['VariableRate'] = (df['FixedOrVariableInterestInd'] == 'V').astype(int)
    df['LongTermNoCollateral'] = ((df['TermInMonths'] >= 180) & (df['CollateralInd'] == 'N')).astype(int)
    return df


def main(trials: int = 120) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / 'data' / 'input'
    out = base_dir / 'data' / 'output'
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / 'train.csv')
    y = df['LoanStatus'].values
    X = df.drop(columns=['LoanStatus'])

    # global stats for features (target-free)
    interest_mean = X['InitialInterestRate'].mean()
    interest_bins = np.quantile(X['InitialInterestRate'], [0, 0.25, 0.5, 0.75, 1.0])
    X = add_features(X, interest_mean, interest_bins)

    # ensure string types for high-card IDs
    X['CongressionalDistrict'] = X['CongressionalDistrict'].astype(str)
    X['RevolverStatus'] = X['RevolverStatus'].astype(str)
    id_col = 'id'
    X = X.drop(columns=[id_col])
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    def objective(trial: optuna.Trial) -> float:
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1200),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
            'random_state': 42,
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'verbose': False,
            'allow_writing_files': False,
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'MVS']),
        }
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.1, 10.0, log=True)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        for tr, va in cv.split(X, y):
            model = CatBoostClassifier(**params)
            model.fit(X.iloc[tr], y[tr], cat_features=cat_cols)
            oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        thresholds = np.linspace(0.0, 1.0, 101)
        scores = [f1_score(y, (oof > t).astype(int)) for t in thresholds]
        return float(np.max(scores))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)

    # re-evaluate best for threshold and save OOF
    best_params = study.best_params
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    model = None
    for tr, va in cv.split(X, y):
        model = CatBoostClassifier(**best_params, random_state=42, loss_function='Logloss', verbose=False, allow_writing_files=False)
        model.fit(X.iloc[tr], y[tr], cat_features=cat_cols)
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    thresholds = np.linspace(0.0, 1.0, 101)
    scores = [f1_score(y, (oof > t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(scores))
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'optuna_cat_best.json').open('w', encoding='utf-8') as f:
        json.dump({'best_params': best_params, 'best_score': float(np.max(scores)), 'best_threshold': float(thresholds[best_idx])}, f, ensure_ascii=False, indent=2)
    np.save(out / 'oof_cat.npy', oof)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=120)
    args = p.parse_args()
    main(trials=args.trials)
