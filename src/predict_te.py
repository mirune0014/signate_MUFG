import json
from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier


def add_features(df: pd.DataFrame,
                 interest_mean: float,
                 interest_bins: np.ndarray,
                 naics_interest_mean: pd.Series,
                 naics_gross_mean: pd.Series,
                 prog_interest_mean: pd.Series,
                 prog_gross_mean: pd.Series,
                 btype_interest_mean: pd.Series,
                 btype_gross_mean: pd.Series,
                 year_interest_mean: pd.Series,
                 year_gross_mean: pd.Series,
                 year_program_interest_mean: pd.Series,
                 year_program_gross_mean: pd.Series,
                 freq_maps: dict) -> pd.DataFrame:
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
    df['CongressionalDistrict'] = df['CongressionalDistrict'].astype(str)
    df['RevolverStatus'] = df['RevolverStatus'].astype(str)
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
    df['SectorProgram'] = df['NaicsSector'] + '_' + df['Subprogram']
    df['SectorBusinessType'] = df['NaicsSector'] + '_' + df['BusinessType']
    df['YearProgram'] = df['ApprovalFiscalYear'].astype(str) + '_' + df['Subprogram']
    df['RateTerm'] = df['InterestRateBucket'] + '_' + df['TermBucket']
    df['HasCollateral'] = (df['CollateralInd'] == 'Y').astype(int)
    df['VariableRate'] = (df['FixedOrVariableInterestInd'] == 'V').astype(int)
    df['LongTermNoCollateral'] = ((df['TermInMonths'] >= 180) & (df['CollateralInd'] == 'N')).astype(int)
    df['AnnualInterestBurden'] = (df['InitialInterestRate'] * df['GrossApproval']) / 100
    df['MonthlyInterestBurden'] = df['AnnualInterestBurden'] / 12
    df['DebtServiceToGuarantee'] = df['MonthlyPayment'] / df['SBAGuaranteedApproval']
    df['SectorInterestDiff'] = df['InitialInterestRate'] - df['NaicsSector'].map(naics_interest_mean)
    df['SectorApprovalDiff'] = df['GrossApproval'] - df['NaicsSector'].map(naics_gross_mean)
    df['ProgramInterestDiff'] = df['InitialInterestRate'] - df['Subprogram'].map(prog_interest_mean)
    df['ProgramApprovalDiff'] = df['GrossApproval'] - df['Subprogram'].map(prog_gross_mean)
    df['BTypeInterestDiff'] = df['InitialInterestRate'] - df['BusinessType'].map(btype_interest_mean)
    df['BTypeApprovalDiff'] = df['GrossApproval'] - df['BusinessType'].map(btype_gross_mean)
    df['YearInterestDiff'] = df['InitialInterestRate'] - df['ApprovalFiscalYear'].map(year_interest_mean)
    df['YearApprovalDiff'] = df['GrossApproval'] - df['ApprovalFiscalYear'].map(year_gross_mean)
    yp_int = year_program_interest_mean.to_dict()
    yp_gross = year_program_gross_mean.to_dict()
    keys = list(zip(df['ApprovalFiscalYear'], df['Subprogram']))
    df['YearProgramInterestDiff'] = df['InitialInterestRate'] - pd.Series([yp_int.get(k, np.nan) for k in keys], index=df.index)
    df['YearProgramApprovalDiff'] = df['GrossApproval'] - pd.Series([yp_gross.get(k, np.nan) for k in keys], index=df.index)
    for col, fmap in freq_maps.items():
        df[f'Freq_{col}'] = df[col].map(fmap).astype(float)
    year_med = int(df['ApprovalFiscalYear'].median())
    df['YearCentered'] = df['ApprovalFiscalYear'] - year_med
    df['Post2010'] = (df['ApprovalFiscalYear'] >= 2010).astype(int)
    df['Post2015'] = (df['ApprovalFiscalYear'] >= 2015).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "input"
    output_dir = base_dir / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')

    target_col = 'LoanStatus'
    id_col = 'id'

    # Precompute global stats
    interest_mean = train['InitialInterestRate'].mean()
    interest_bins = np.quantile(train['InitialInterestRate'], [0, 0.25, 0.5, 0.75, 1.0])
    naics_interest_mean = train.groupby('NaicsSector')['InitialInterestRate'].mean()
    naics_gross_mean = train.groupby('NaicsSector')['GrossApproval'].mean()
    prog_interest_mean = train.groupby('Subprogram')['InitialInterestRate'].mean()
    prog_gross_mean = train.groupby('Subprogram')['GrossApproval'].mean()
    btype_interest_mean = train.groupby('BusinessType')['InitialInterestRate'].mean()
    btype_gross_mean = train.groupby('BusinessType')['GrossApproval'].mean()
    year_interest_mean = train.groupby('ApprovalFiscalYear')['InitialInterestRate'].mean()
    year_gross_mean = train.groupby('ApprovalFiscalYear')['GrossApproval'].mean()
    year_program_interest_mean = train.groupby(['ApprovalFiscalYear','Subprogram'])['InitialInterestRate'].mean()
    year_program_gross_mean = train.groupby(['ApprovalFiscalYear','Subprogram'])['GrossApproval'].mean()
    freq_maps = {
        'NaicsSector': train['NaicsSector'].value_counts(),
        'BusinessType': train['BusinessType'].value_counts(),
        'Subprogram': train['Subprogram'].value_counts(),
        'FixedOrVariableInterestInd': train['FixedOrVariableInterestInd'].value_counts(),
        'RevolverStatus': train['RevolverStatus'].astype(str).value_counts(),
        'CollateralInd': train['CollateralInd'].value_counts(),
        'CongressionalDistrict': train['CongressionalDistrict'].astype(str).value_counts(),
        'ApprovalFiscalYear': train['ApprovalFiscalYear'].value_counts(),
        'YearProgram': (train['ApprovalFiscalYear'].astype(str) + '_' + train['Subprogram']).value_counts(),
    }

    X_train = add_features(
        train.drop(columns=[target_col]),
        interest_mean, interest_bins,
        naics_interest_mean, naics_gross_mean,
        prog_interest_mean, prog_gross_mean,
        btype_interest_mean, btype_gross_mean,
        year_interest_mean, year_gross_mean,
        year_program_interest_mean, year_program_gross_mean,
        freq_maps,
    )
    y_train = train[target_col].values
    X_test = add_features(
        test.copy(), interest_mean, interest_bins,
        naics_interest_mean, naics_gross_mean,
        prog_interest_mean, prog_gross_mean,
        btype_interest_mean, btype_gross_mean,
        year_interest_mean, year_gross_mean,
        year_program_interest_mean, year_program_gross_mean,
        freq_maps,
    )

    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    X_train = X_train.drop(columns=[id_col])
    X_test_ids = X_test[id_col].copy()
    X_test = X_test.drop(columns=[id_col])

    # best params if available (TE-specific has priority)
    best_params = {}
    te_cfg = output_dir / 'optuna_te_best.json'
    if te_cfg.exists():
        with te_cfg.open('r', encoding='utf-8') as f:
            best_params = json.load(f).get('best_params', {})
    else:
        params_path = output_dir / 'optuna_best_params.json'
        if params_path.exists():
            with params_path.open('r', encoding='utf-8') as f:
                best_params = json.load(f).get('best_params', {})

    # target encoding on full training (final fit)
    if 'te_smoothing' in best_params or 'te_min_samples_leaf' in best_params:
        te = TargetEncoder(
            cols=cat_cols,
            smoothing=best_params.get('te_smoothing', 1.0),
            min_samples_leaf=best_params.get('te_min_samples_leaf', 1),
        )
    else:
        te = TargetEncoder(cols=cat_cols)
    X_train_enc = te.fit_transform(X_train, y_train)
    X_test_enc = te.transform(X_test)

    # imbalance handling
    pos = int((y_train == 1).sum())
    neg = len(y_train) - pos
    spw = neg / max(pos, 1)

    params = {
        'objective': 'binary',
        'random_state': 42,
        'n_estimators': best_params.get('n_estimators', 200),
        'learning_rate': best_params.get('learning_rate', 0.1),
        'num_leaves': best_params.get('num_leaves', 31),
        'max_depth': best_params.get('max_depth', -1),
        'min_child_samples': best_params.get('min_data_in_leaf', 20),
        'feature_fraction': best_params.get('feature_fraction', 1.0),
        'bagging_fraction': best_params.get('bagging_fraction', 1.0),
        'lambda_l1': best_params.get('lambda_l1', 0.0),
        'lambda_l2': best_params.get('lambda_l2', 0.0),
        'scale_pos_weight': best_params.get('scale_pos_weight', spw),
    }

    model = LGBMClassifier(**params)
    model.fit(X_train_enc, y_train)
    prob = model.predict_proba(X_test_enc)[:, 1]

    # threshold preference: TE-optimized > ensemble > baseline threshold
    thr = 0.5
    te_thr_path = output_dir / 'optuna_te_best.json'
    if te_thr_path.exists():
        try:
            with te_thr_path.open('r', encoding='utf-8') as f:
                tecfg = json.load(f)
                thr = float(tecfg.get('best_threshold', thr))
        except Exception:
            pass
    else:
        for p in ['ensemble_results.json', 'threshold_results.json']:
            path = output_dir / p
            if path.exists():
                try:
                    with path.open('r', encoding='utf-8') as f:
                        d = json.load(f)
                        thr = float(d.get('best_threshold', thr))
                        break
                except Exception:
                    pass

    pred = (prob > thr).astype(int)
    submission = pd.DataFrame({'Id': X_test_ids, 'LoanStatus': pred})
    submission.to_csv(output_dir / 'submit_te.csv', index=False, header=False)


if __name__ == '__main__':
    main()
