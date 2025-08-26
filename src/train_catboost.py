import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in cv.split(X, y):
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function='Logloss',
            verbose=False,
            random_state=42,
        )
        model.fit(X.iloc[train_idx], y[train_idx], cat_features=cat_cols)
        preds = (model.predict_proba(X.iloc[valid_idx])[:, 1] > 0.5).astype(int)
        scores.append(f1_score(y[valid_idx], preds))

    results = {"f1_scores": scores, "mean_f1": float(np.mean(scores))}
    output_dir = Path(r"C:\Users\miots\ruruprojects3\MUFG\signate_MUFG\data\output")
    with open(output_dir / 'catboost_cv_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
