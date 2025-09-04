import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    interest_mean = df['InitialInterestRate'].mean()
    interest_bins = np.quantile(df['InitialInterestRate'], [0, 0.25, 0.5, 0.75, 1.0])
    business_age_map = {
        'Startup, Loan Funds will Open Business': 0,
        'New Business or 2 years or less': 2,
        'Existing or more than 2 years old': 5,
        'Change of Ownership': 5,
        'Unanswered': -1,
    }
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
    return df


def main(n_trials: int = 50):
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / 'data' / 'input'
    out = base_dir / 'data' / 'output'
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / 'train.csv')
    target_col = 'LoanStatus'
    id_col = 'id'

    X = add_features(train.drop(columns=[target_col]))
    y = train[target_col].values
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    X = X.drop(columns=[id_col])

    pos = int((y == 1).sum())
    neg = len(y) - pos
    base_spw = neg / max(pos, 1)

    def objective(trial: optuna.Trial) -> float:
        # TE params
        te_smoothing = trial.suggest_float('te_smoothing', 0.1, 50.0, log=True)
        te_min_samples = trial.suggest_int('te_min_samples_leaf', 1, 50)
        # LGBM params
        params = {
            'objective': 'binary',
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 64),
            'max_depth': trial.suggest_int('max_depth', -1, 16),
            'min_child_samples': trial.suggest_int('min_data_in_leaf', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', base_spw * 0.5, base_spw * 1.5),
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        for tr, va in cv.split(X, y):
            te = TargetEncoder(cols=cat_cols, smoothing=te_smoothing, min_samples_leaf=te_min_samples)
            Xtr = te.fit_transform(X.iloc[tr], y[tr])
            Xva = te.transform(X.iloc[va])
            model = LGBMClassifier(**params)
            model.fit(Xtr, y[tr])
            oof[va] = model.predict_proba(Xva)[:, 1]
        # threshold sweep on OOF
        thresholds = np.linspace(0.0, 1.0, 101)
        scores = [f1_score(y, (oof > t).astype(int)) for t in thresholds]
        return float(np.max(scores))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Re-evaluate best to compute best threshold and persist
    best_params = study.best_params
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    te_smoothing = best_params.get('te_smoothing', 1.0)
    te_min_samples = best_params.get('te_min_samples_leaf', 1)
    for tr, va in cv.split(X, y):
        te = TargetEncoder(cols=cat_cols, smoothing=te_smoothing, min_samples_leaf=te_min_samples)
        Xtr = te.fit_transform(X.iloc[tr], y[tr])
        Xva = te.transform(X.iloc[va])
        model = LGBMClassifier(
            objective='binary', random_state=42,
            n_estimators=best_params.get('n_estimators', 200),
            learning_rate=best_params.get('learning_rate', 0.1),
            num_leaves=best_params.get('num_leaves', 31),
            max_depth=best_params.get('max_depth', -1),
            min_child_samples=best_params.get('min_data_in_leaf', 20),
            feature_fraction=best_params.get('feature_fraction', 1.0),
            bagging_fraction=best_params.get('bagging_fraction', 1.0),
            lambda_l1=best_params.get('lambda_l1', 0.0),
            lambda_l2=best_params.get('lambda_l2', 0.0),
            scale_pos_weight=best_params.get('scale_pos_weight', base_spw),
        )
        model.fit(Xtr, y[tr])
        oof[va] = model.predict_proba(Xva)[:, 1]
    thresholds = np.linspace(0.0, 1.0, 101)
    scores = [f1_score(y, (oof > t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(scores))
    best = {
        'best_params': best_params,
        'best_score': float(np.max(scores)),
        'best_threshold': float(thresholds[best_idx]),
    }
    with (out / 'optuna_te_best.json').open('w', encoding='utf-8') as f:
        json.dump(best, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=50)
    args = p.parse_args()
    main(n_trials=args.trials)
