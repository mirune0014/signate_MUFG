import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "input"
    output_dir = base_dir / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')

    target_col = 'LoanStatus'
    id_col = 'id'

    # check missing values
    if train.isnull().values.any() or test.isnull().values.any():
        raise ValueError('Missing values found in datasets')


    # --- feature engineering (global stats from train only; no target used) ---
    interest_mean = train['InitialInterestRate'].mean()
    interest_bins = np.quantile(train['InitialInterestRate'], [0, 0.25, 0.5, 0.75, 1.0])
    business_age_map = {
        'Startup, Loan Funds will Open Business': 0,
        'New Business or 2 years or less': 2,
        'Existing or more than 2 years old': 5,
        'Change of Ownership': 5,
        'Unanswered': -1,
    }
    # group means
    naics_interest_mean = train.groupby('NaicsSector')['InitialInterestRate'].mean()
    naics_gross_mean = train.groupby('NaicsSector')['GrossApproval'].mean()
    prog_interest_mean = train.groupby('Subprogram')['InitialInterestRate'].mean()
    prog_gross_mean = train.groupby('Subprogram')['GrossApproval'].mean()
    btype_interest_mean = train.groupby('BusinessType')['InitialInterestRate'].mean()
    btype_gross_mean = train.groupby('BusinessType')['GrossApproval'].mean()
    year_interest_mean = train.groupby('ApprovalFiscalYear')['InitialInterestRate'].mean()
    year_gross_mean = train.groupby('ApprovalFiscalYear')['GrossApproval'].mean()
    year_program_gross_mean = train.groupby(['ApprovalFiscalYear', 'Subprogram'])['GrossApproval'].mean()
    year_program_interest_mean = train.groupby(['ApprovalFiscalYear', 'Subprogram'])['InitialInterestRate'].mean()
    # frequencies
    freq_sector = train['NaicsSector'].value_counts()
    freq_btype = train['BusinessType'].value_counts()
    freq_prog = train['Subprogram'].value_counts()
    # combo frequency to capture interactions without OHE explosion
    combo_prog_btype = (train['Subprogram'] + '_' + train['BusinessType']).value_counts()
    freq_fixvar = train['FixedOrVariableInterestInd'].value_counts()
    freq_revolver = train['RevolverStatus'].value_counts()
    freq_collateral = train['CollateralInd'].value_counts()
    freq_year = train['ApprovalFiscalYear'].value_counts()
    freq_cd = train['CongressionalDistrict'].value_counts()
    combo_year_prog = (train['ApprovalFiscalYear'].astype(str) + '_' + train['Subprogram']).value_counts()

    def safe_div(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(b == 0, np.nan, a / b)
        return out

    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # base transforms
        df['GrossApproval_log'] = np.log1p(df['GrossApproval'])
        df['SBAGuaranteedApproval_log'] = np.log1p(df['SBAGuaranteedApproval'])
        df['TermInYears'] = safe_div(df['TermInMonths'], 12)
        df['MonthlyPayment'] = safe_div(df['GrossApproval'], df['TermInMonths'])
        df['MonthlyPayment_log'] = np.log1p(df['MonthlyPayment'])
        df['ApprovalRatio'] = safe_div(df['SBAGuaranteedApproval'], df['GrossApproval'])
        df['ApprovalDiff'] = df['GrossApproval'] - df['SBAGuaranteedApproval']
        df['GuaranteeGapRatio'] = safe_div(df['ApprovalDiff'], df['GrossApproval'])
        df['InterestRateDiff'] = df['InitialInterestRate'] - interest_mean
        df['InterestRateBucket'] = np.digitize(
            df['InitialInterestRate'], interest_bins[1:-1], right=True
        ).astype(str)
        df['BusinessAgeNum'] = df['BusinessAge'].map(business_age_map)
        df['InterestRatePerTerm'] = safe_div(df['InitialInterestRate'], df['TermInMonths'])
        df['JobsSupportedRatio'] = safe_div(df['JobsSupported'], df['GrossApproval'])
        df['Jobs_log'] = np.log1p(df['JobsSupported'])
        df['GrossPerJob'] = safe_div(df['GrossApproval'], df['JobsSupported'].replace(0, np.nan))
        df['GuaranteePerJob'] = safe_div(df['SBAGuaranteedApproval'], df['JobsSupported'].replace(0, np.nan))
        df['TermBucket'] = pd.cut(
            df['TermInMonths'], bins=[0, 60, 120, 180, 240, 360], labels=False, include_lowest=True
        ).astype(str)
        df['ProgramBusinessType'] = df['Subprogram'] + '_' + df['BusinessType']
        # interactions and risk flags
        df['RateTerm'] = df['InterestRateBucket'] + '_' + df['TermBucket']
        df['HasCollateral'] = (df['CollateralInd'] == 'Y').astype(int)
        df['VariableRate'] = (df['FixedOrVariableInterestInd'] == 'V').astype(int)
        df['LongTermNoCollateral'] = ((df['TermInMonths'] >= 180) & (df['CollateralInd'] == 'N')).astype(int)
        # interest burden approximations (InitialInterestRate is in percent)
        df['AnnualInterestBurden'] = safe_div(df['InitialInterestRate'] * df['GrossApproval'], 100)
        df['MonthlyInterestBurden'] = safe_div(df['AnnualInterestBurden'], 12)
        df['DebtServiceToGuarantee'] = safe_div(df['MonthlyPayment'], df['SBAGuaranteedApproval'])
        # group deltas
        df['SectorInterestDiff'] = df['InitialInterestRate'] - df['NaicsSector'].map(naics_interest_mean)
        df['SectorApprovalDiff'] = df['GrossApproval'] - df['NaicsSector'].map(naics_gross_mean)
        df['ProgramInterestDiff'] = df['InitialInterestRate'] - df['Subprogram'].map(prog_interest_mean)
        df['ProgramApprovalDiff'] = df['GrossApproval'] - df['Subprogram'].map(prog_gross_mean)
        df['BTypeInterestDiff'] = df['InitialInterestRate'] - df['BusinessType'].map(btype_interest_mean)
        df['BTypeApprovalDiff'] = df['GrossApproval'] - df['BusinessType'].map(btype_gross_mean)
        df['YearInterestDiff'] = df['InitialInterestRate'] - df['ApprovalFiscalYear'].map(year_interest_mean)
        df['YearApprovalDiff'] = df['GrossApproval'] - df['ApprovalFiscalYear'].map(year_gross_mean)
        # year-program means (map via tuple keys)
        yp_gross = year_program_gross_mean.to_dict()
        yp_int = year_program_interest_mean.to_dict()
        keys = list(zip(df['ApprovalFiscalYear'], df['Subprogram']))
        df['YearProgramApprovalDiff'] = df['GrossApproval'] - pd.Series([yp_gross.get(k, np.nan) for k in keys], index=df.index)
        df['YearProgramInterestDiff'] = df['InitialInterestRate'] - pd.Series([yp_int.get(k, np.nan) for k in keys], index=df.index)
        # frequencies
        df['Freq_NaicsSector'] = df['NaicsSector'].map(freq_sector).astype(float)
        df['Freq_BusinessType'] = df['BusinessType'].map(freq_btype).astype(float)
        df['Freq_Subprogram'] = df['Subprogram'].map(freq_prog).astype(float)
        df['Freq_ProgramBusinessType'] = (df['Subprogram'] + '_' + df['BusinessType']).map(combo_prog_btype).astype(float)
        df['Freq_FixedVar'] = df['FixedOrVariableInterestInd'].map(freq_fixvar).astype(float)
        df['Freq_Revolver'] = df['RevolverStatus'].map(freq_revolver).astype(float)
        df['Freq_Collateral'] = df['CollateralInd'].map(freq_collateral).astype(float)
        df['Freq_Year'] = df['ApprovalFiscalYear'].map(freq_year).astype(float)
        df['Freq_CongressionalDistrict'] = df['CongressionalDistrict'].map(freq_cd).astype(float)
        df['Freq_YearProgram'] = (df['ApprovalFiscalYear'].astype(str) + '_' + df['Subprogram']).map(combo_year_prog).astype(float)
        # simple risk flags & temporal signals
        df['HighInterestFlag'] = (df['InitialInterestRate'] >= np.quantile(train['InitialInterestRate'], 0.75)).astype(int)
        df['LongTermFlag'] = (df['TermInMonths'] >= 180).astype(int)
        year_med = int(train['ApprovalFiscalYear'].median())
        df['YearCentered'] = df['ApprovalFiscalYear'] - year_med
        df['Post2010'] = (df['ApprovalFiscalYear'] >= 2010).astype(int)
        df['Post2015'] = (df['ApprovalFiscalYear'] >= 2015).astype(int)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
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
