import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    data_dir = Path(r'C:\Users\miots\ruruprojects3\MUFG\signate_MUFG\data\input')
    output_dir = Path(r'C:\Users\miots\ruruprojects3\MUFG\signate_MUFG\data\output')
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')

    target_col = 'LoanStatus'
    id_col = 'id'

    # check missing values
    if train.isnull().values.any() or test.isnull().values.any():
        raise ValueError('Missing values found in datasets')


    # --- feature engineering ---
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

    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
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
        return df

    X_train = add_features(train.drop(columns=[target_col]))
    y_train = train[target_col]
    X_test = add_features(test.copy())


    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols + [id_col]]

    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    X_train_proc = preprocess.fit_transform(X_train.drop(columns=[id_col]))
    X_test_proc = preprocess.transform(X_test.drop(columns=[id_col]))

    np.savez_compressed(output_dir / 'train_preprocessed.npz', X=X_train_proc, y=y_train.values)
    np.savez_compressed(output_dir / 'test_preprocessed.npz', X=X_test_proc)

    info = {
        'categorical_features': cat_cols,
        'numerical_features': num_cols,
        'n_train': X_train_proc.shape[0],
        'n_features': X_train_proc.shape[1]
    }
    with open(output_dir / 'preprocess_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
