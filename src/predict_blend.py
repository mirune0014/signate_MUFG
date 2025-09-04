import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier


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


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / 'data' / 'input'
    out = base_dir / 'data' / 'output'

    with (out / 'ensemble_results.json').open('r', encoding='utf-8') as f:
        cfg = json.load(f)
    names = cfg['models']
    weights = np.array(cfg['weights'], dtype=float)
    thr = float(cfg.get('best_threshold', 0.5))

    # preprocessed models
    train_npz = np.load(out / 'train_preprocessed.npz')
    X_train = train_npz['X']
    y_train = train_npz['y']
    test_npz = np.load(out / 'test_preprocessed.npz')
    X_test = test_npz['X']

    pos = int((y_train == 1).sum())
    neg = len(y_train) - pos
    spw = neg / max(pos, 1)

    preds = {}

    if 'lgb' in names:
        params = {'objective': 'binary', 'metric': 'None', 'verbose': -1, 'scale_pos_weight': spw}
        model = lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=100)
        preds['lgb'] = model.predict(X_test)

    if 'xgb' in names:
        model = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', n_jobs=-1, random_state=42, verbosity=0, scale_pos_weight=spw
        )
        model.fit(X_train, y_train)
        preds['xgb'] = model.predict_proba(X_test)[:, 1]

    # TE and Cat models need raw features
    if 'te' in names or 'cat' in names:
        train_df = pd.read_csv(data_dir / 'train.csv')
        test_df = pd.read_csv(data_dir / 'test.csv')
        Xtr = add_features(train_df.drop(columns=['LoanStatus']))
        ytr = train_df['LoanStatus'].values
        Xts = add_features(test_df.copy())
        id_col = 'id'
        cat_cols = Xtr.select_dtypes(include='object').columns.tolist()
        Xtr = Xtr.drop(columns=[id_col])
        Xts = Xts.drop(columns=[id_col])
        # TE branch
        if 'te' in names:
            te_params = {}
            te_cfg = out / 'optuna_te_best.json'
            if te_cfg.exists():
                with te_cfg.open('r', encoding='utf-8') as f:
                    te_params = json.load(f).get('best_params', {})
            if 'te_smoothing' in te_params or 'te_min_samples_leaf' in te_params:
                te = TargetEncoder(
                    cols=cat_cols,
                    smoothing=te_params.get('te_smoothing', 1.0),
                    min_samples_leaf=te_params.get('te_min_samples_leaf', 1),
                )
            else:
                te = TargetEncoder(cols=cat_cols)
            Xtr_enc = te.fit_transform(Xtr, ytr)
            Xts_enc = te.transform(Xts)
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                objective='binary', random_state=42,
                n_estimators=te_params.get('n_estimators', 200),
                learning_rate=te_params.get('learning_rate', 0.1),
                num_leaves=te_params.get('num_leaves', 31),
                max_depth=te_params.get('max_depth', -1),
                min_child_samples=te_params.get('min_data_in_leaf', 20),
                feature_fraction=te_params.get('feature_fraction', 1.0),
                bagging_fraction=te_params.get('bagging_fraction', 1.0),
                lambda_l1=te_params.get('lambda_l1', 0.0),
                lambda_l2=te_params.get('lambda_l2', 0.0),
                scale_pos_weight=te_params.get('scale_pos_weight', spw),
            )
            model.fit(Xtr_enc, ytr)
            preds['te'] = model.predict_proba(Xts_enc)[:, 1]
        # CatBoost branch
        if 'cat' in names:
            cat_model = CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=False, random_state=42
            )
            cat_model.fit(Xtr, ytr, cat_features=cat_cols)
            preds['cat'] = cat_model.predict_proba(Xts)[:, 1]

    # blend according to learned weights
    P = np.vstack([preds[n] for n in names]).T
    prob = (P * weights).sum(axis=1)
    pred = (prob > thr).astype(int)

    test_ids = pd.read_csv(data_dir / 'test.csv')['id']
    submission = pd.DataFrame({'Id': test_ids, 'LoanStatus': pred})
    submission.to_csv(out / 'submit_blend.csv', index=False, header=False)


if __name__ == '__main__':
    main()
