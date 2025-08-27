import json
from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders import LeaveOneOutEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class AddFeatures(BaseEstimator, TransformerMixin):
    """Custom transformer that adds features using train-only statistics."""

    def __init__(self, id_col: str = "id"):
        self.id_col = id_col

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X.copy()
        df['__target__'] = y

        self.interest_mean_ = df['InitialInterestRate'].mean()
        self.interest_bins_ = np.quantile(df['InitialInterestRate'], [0, 0.25, 0.5, 0.75, 1.0])
        self.business_age_map_ = {
            'Startup, Loan Funds will Open Business': 0,
            'New Business or 2 years or less': 2,
            'Existing or more than 2 years old': 5,
            'Change of Ownership': 5,
            'Unanswered': -1,
        }
        self.naics_interest_mean_ = df.groupby('NaicsSector')['InitialInterestRate'].mean()
        self.naics_gross_mean_ = df.groupby('NaicsSector')['GrossApproval'].mean()
        self.program_default_ = df.groupby('Subprogram')['__target__'].mean()
        self.sector_default_ = df.groupby('NaicsSector')['__target__'].mean()
        self.program_gross_mean_ = df.groupby('Subprogram')['GrossApproval'].mean()
        self.sector_term_mean_ = df.groupby('NaicsSector')['TermInMonths'].mean()
        self.program_term_mean_ = df.groupby('Subprogram')['TermInMonths'].mean()
        self.overall_default_ = df['__target__'].mean()
        self.gross_bins_ = pd.qcut(df['GrossApproval'], q=5, retbins=True, duplicates='drop')[1]
        self.term_bins_ = pd.qcut(df['TermInMonths'], q=5, retbins=True, duplicates='drop')[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df['GrossApproval_log'] = np.log1p(df['GrossApproval'])
        df['SBAGuaranteedApproval_log'] = np.log1p(df['SBAGuaranteedApproval'])
        df['MonthlyPayment'] = df['GrossApproval'] / df['TermInMonths']
        df['ApprovalRatio'] = df['SBAGuaranteedApproval'] / df['GrossApproval']
        df['ApprovalDiff'] = df['GrossApproval'] - df['SBAGuaranteedApproval']
        df['InterestRateDiff'] = df['InitialInterestRate'] - self.interest_mean_
        df['InterestRateBucket'] = np.digitize(
            df['InitialInterestRate'], self.interest_bins_[1:-1], right=True
        ).astype(str)
        df['BusinessAgeNum'] = df['BusinessAge'].map(self.business_age_map_)
        df['InterestRatePerTerm'] = df['InitialInterestRate'] / df['TermInMonths']
        df['JobsSupportedRatio'] = df['JobsSupported'] / df['GrossApproval']
        df['TermBucket'] = pd.cut(
            df['TermInMonths'], bins=[0, 60, 120, 180, 240, 360], labels=False, include_lowest=True
        ).astype(str)
        df['ProgramBusinessType'] = df['Subprogram'] + '_' + df['BusinessType']
        df['SectorInterestDiff'] = df['InitialInterestRate'] - df['NaicsSector'].map(self.naics_interest_mean_)
        df['SectorApprovalDiff'] = df['GrossApproval'] - df['NaicsSector'].map(self.naics_gross_mean_)
        df['ProgramDefaultDiff'] = df['Subprogram'].map(self.program_default_) - self.overall_default_
        df['SectorDefaultDiff'] = df['NaicsSector'].map(self.sector_default_) - self.overall_default_
        df['ProgramLoanDiff'] = df['GrossApproval'] - df['Subprogram'].map(self.program_gross_mean_)
        df['SectorTermDiff'] = df['TermInMonths'] - df['NaicsSector'].map(self.sector_term_mean_)
        df['ProgramTermDiff'] = df['TermInMonths'] - df['Subprogram'].map(self.program_term_mean_)
        df['SectorProgram'] = df['NaicsSector'] + '_' + df['Subprogram']
        df['GrossApprovalBin'] = pd.cut(
            df['GrossApproval'], bins=self.gross_bins_, labels=False, include_lowest=True
        ).astype(str)
        df['TermInMonthsBin'] = pd.cut(
            df['TermInMonths'], bins=self.term_bins_, labels=False, include_lowest=True
        ).astype(str)
        return df.drop(columns=[self.id_col])


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "input"
    output_dir = base_dir / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / 'train.csv')
    target_col = 'LoanStatus'
    X = train.drop(columns=[target_col])
    y = train[target_col]

    preprocess = ColumnTransformer(
        transformers=[
            ('cat', LeaveOneOutEncoder(), make_column_selector(dtype_include=object)),
            ('num', StandardScaler(), make_column_selector(dtype_exclude=object)),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42,
    )

    pipe = Pipeline([
        ('fe', AddFeatures()),
        ('preprocess', preprocess),
        ('model', model),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in cv.split(X, y):
        pipe_clone = Pipeline(pipe.steps)  # fresh pipeline per fold
        pipe_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = pipe_clone.predict_proba(X.iloc[valid_idx])[:, 1]
        preds = (probs >= 0.3).astype(int)
        scores.append(f1_score(y.iloc[valid_idx], preds))

    results = {"f1_scores": scores, "mean_f1": float(np.mean(scores))}
    with open(output_dir / 'pipeline_cv_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
