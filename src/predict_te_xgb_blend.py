import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier
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
    # cast discrete identifiers to categorical strings for TE consistency
    df['CongressionalDistrict'] = df['CongressionalDistrict'].astype(str)
    df['RevolverStatus'] = df['RevolverStatus'].astype(str)
    df['MonthlyPayment'] = df['GrossApproval'] / df['TermInMonths']
    df['MonthlyPayment_log'] = np.log1p(df['MonthlyPayment'])
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
    # group deltas
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


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / 'data' / 'input'
    out = base_dir / 'data' / 'output'

    # load blend config
    with (out / 'ensemble_te_xgb.json').open('r', encoding='utf-8') as f:
        cfg = json.load(f)
    if 'weights' in cfg:
        weights = cfg['weights']
        thr = float(cfg['threshold'])
    else:
        w_te = float(cfg['weight_te'])
        thr = float(cfg['threshold'])

    # TE probabilities
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    target_col = 'LoanStatus'
    id_col = 'id'
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
    Xtr = add_features(train.drop(columns=[target_col]), interest_mean, interest_bins,
                       naics_interest_mean, naics_gross_mean, prog_interest_mean, prog_gross_mean,
                       btype_interest_mean, btype_gross_mean, year_interest_mean, year_gross_mean,
                       year_program_interest_mean, year_program_gross_mean, freq_maps)
    ytr = train[target_col].values
    Xts = add_features(test.copy(), interest_mean, interest_bins,
                       naics_interest_mean, naics_gross_mean, prog_interest_mean, prog_gross_mean,
                       btype_interest_mean, btype_gross_mean, year_interest_mean, year_gross_mean,
                       year_program_interest_mean, year_program_gross_mean, freq_maps)
    cat_cols = Xtr.select_dtypes(include='object').columns.tolist()
    Xtr = Xtr.drop(columns=[id_col])
    Xts = Xts.drop(columns=[id_col])

    te_params = {}
    te_cfg_path = out / 'optuna_te_best.json'
    if te_cfg_path.exists():
        with te_cfg_path.open('r', encoding='utf-8') as f:
            te_params = json.load(f).get('best_params', {})
    if 'te_smoothing' in te_params or 'te_min_samples_leaf' in te_params:
        te = TargetEncoder(cols=cat_cols,
                           smoothing=te_params.get('te_smoothing', 1.0),
                           min_samples_leaf=te_params.get('te_min_samples_leaf', 1))
    else:
        te = TargetEncoder(cols=cat_cols)
    Xtr_enc = te.fit_transform(Xtr, ytr)
    Xts_enc = te.transform(Xts)
    seeds = [42, 1337, 2025]
    te_probs = []
    for sd in seeds:
        te_model = LGBMClassifier(
            objective='binary', random_state=sd,
            n_estimators=te_params.get('n_estimators', 200),
            learning_rate=te_params.get('learning_rate', 0.1),
            num_leaves=te_params.get('num_leaves', 31),
            max_depth=te_params.get('max_depth', -1),
            min_child_samples=te_params.get('min_data_in_leaf', 20),
            feature_fraction=te_params.get('feature_fraction', 1.0),
            bagging_fraction=te_params.get('bagging_fraction', 1.0),
            lambda_l1=te_params.get('lambda_l1', 0.0),
            lambda_l2=te_params.get('lambda_l2', 0.0),
            scale_pos_weight=te_params.get('scale_pos_weight', (len(ytr)-ytr.sum())/max(ytr.sum(),1)),
        )
        te_model.fit(Xtr_enc, ytr)
        te_probs.append(te_model.predict_proba(Xts_enc)[:, 1])
    te_prob = np.mean(np.stack(te_probs, axis=0), axis=0)

    # XGB probabilities (preprocessed)
    train_npz = np.load(out / 'train_preprocessed.npz')
    X_pre = train_npz['X']
    y_pre = train_npz['y']
    X_pre_test = np.load(out / 'test_preprocessed.npz')['X']

    tuned = {}
    xgb_cfg = out / 'optuna_xgb_best.json'
    if xgb_cfg.exists():
        with xgb_cfg.open('r', encoding='utf-8') as f:
            tuned = json.load(f).get('best_params', {})
    pos = int((y_pre == 1).sum()); neg = len(y_pre)-pos
    base = dict(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', n_jobs=-1, random_state=42, verbosity=0, scale_pos_weight=neg/max(pos,1))
    base.update(tuned)
    xgb_model = xgb.XGBClassifier(**base)
    xgb_model.fit(X_pre, y_pre)
    xgb_prob = xgb_model.predict_proba(X_pre_test)[:, 1]

    # CatBoost probabilities (train full, predict test) if blending needs it
    cat_prob = None
    if isinstance(cfg, dict) and ('weights' in cfg or cfg.get('models', None)):
        try:
            tuned_cat = {}
            cat_cfg = out / 'optuna_cat_best.json'
            if cat_cfg.exists():
                with cat_cfg.open('r', encoding='utf-8') as f:
                    tuned_cat = json.load(f).get('best_params', {})
            # light bagging over seeds for stability
            seeds = [42, 1337, 2025]
            probs = []
            for sd in seeds:
                cb = CatBoostClassifier(
                    iterations=tuned_cat.get('iterations', 700),
                    depth=tuned_cat.get('depth', 6),
                    learning_rate=tuned_cat.get('learning_rate', 0.1),
                    l2_leaf_reg=tuned_cat.get('l2_leaf_reg', 3.0),
                    bootstrap_type=tuned_cat.get('bootstrap_type', 'Bayesian'),
                    bagging_temperature=tuned_cat.get('bagging_temperature', 1.0),
                    loss_function='Logloss',
                    verbose=False,
                    random_state=sd,
                )
                cb.fit(Xtr, ytr, cat_features=cat_cols)
                probs.append(cb.predict_proba(Xts)[:, 1])
            cat_prob = np.mean(np.stack(probs, axis=0), axis=0)
        except Exception:
            cat_prob = None

    # blend
    if 'weights' in cfg and cat_prob is not None:
        w_te, w_xgb, w_cat = float(weights[0]), float(weights[1]), float(weights[2])
        s = w_te + w_xgb + w_cat
        if s <= 0:
            w_te, w_xgb, w_cat = 1/3, 1/3, 1/3
        else:
            w_te, w_xgb, w_cat = w_te/s, w_xgb/s, w_cat/s
        prob = w_te * te_prob + w_xgb * xgb_prob + w_cat * cat_prob
    else:
        prob = w_te * te_prob + (1 - w_te) * xgb_prob
    # threshold preference: time-aware > ensemble > default 0.5
    time_thr_path = out / 'threshold_time_results.json'
    if time_thr_path.exists():
        try:
            with time_thr_path.open('r', encoding='utf-8') as f:
                thr = float(json.load(f).get('best_threshold', thr))
        except Exception:
            pass
    pred = (prob > thr).astype(int)
    test_ids = pd.read_csv(data_dir / 'test.csv')['id']
    sub = pd.DataFrame({'Id': test_ids, 'LoanStatus': pred})
    sub.to_csv(out / 'submit_te_xgb.csv', index=False, header=False)

    # persist blended probabilities for downstream thresholding experiments
    np.save(out / 'test_blend_prob.npy', prob)
    np.save(out / 'test_prob_te.npy', te_prob)
    np.save(out / 'test_prob_xgb.npy', xgb_prob)
    if cat_prob is not None:
        np.save(out / 'test_prob_cat.npy', cat_prob)


if __name__ == '__main__':
    main()
