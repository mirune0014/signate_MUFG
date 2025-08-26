import json
from pathlib import Path

import numpy as np
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


def main():

    data_dir = Path(r"C:\Users\miots\ruruprojects3\MUFG\signate_MUFG\data\input")

    train = pd.read_csv(data_dir / 'train.csv')
    target_col = 'LoanStatus'
    id_col = 'id'

    X = add_features(train.drop(columns=[target_col]))
    y = train[target_col].values

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    X = X.drop(columns=[id_col])

    params_path = Path('data/output/optuna_best_params.json')
    if params_path.exists():
        with open(params_path, 'r', encoding='utf-8') as f:
            best_params = json.load(f)['best_params']
    else:
        best_params = {}
    params = {
        'learning_rate': best_params.get('learning_rate', 0.1),
        'num_leaves': best_params.get('num_leaves', 31),
        'max_depth': best_params.get('max_depth', -1),
        'min_child_samples': best_params.get('min_data_in_leaf', 20),
        'feature_fraction': best_params.get('feature_fraction', 1.0),
        'bagging_fraction': best_params.get('bagging_fraction', 1.0),
        'lambda_l1': best_params.get('lambda_l1', 0.0),
        'lambda_l2': best_params.get('lambda_l2', 0.0),
        'scale_pos_weight': best_params.get('scale_pos_weight', 1.0),
        'n_estimators': 100,
        'objective': 'binary',
        'random_state': 42,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in cv.split(X, y):
        te = TargetEncoder(cols=cat_cols)
        X_train = te.fit_transform(X.iloc[train_idx], y[train_idx])
        X_valid = te.transform(X.iloc[valid_idx])
        model = LGBMClassifier(**params)
        model.fit(X_train, y[train_idx])
        preds = model.predict(X_valid)
        scores.append(f1_score(y[valid_idx], preds))

    results = {"f1_scores": scores, "mean_f1": float(np.mean(scores))}

    output_dir = Path(r"C:\Users\miots\ruruprojects3\MUFG\signate_MUFG\data\output")

    with open(output_dir / 'te_cv_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
