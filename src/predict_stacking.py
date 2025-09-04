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

    # load meta config
    with (out / 'stacking_results.json').open('r', encoding='utf-8') as f:
        cfg = json.load(f)
    names = cfg['models']
    coef = np.array(cfg['coef'], dtype=float)
    intercept = float(cfg['intercept'])
    thr = float(cfg.get('best_threshold', 0.5))

    # load train preprocessed for imbalance stats
    train_npz = np.load(out / 'train_preprocessed.npz')
    X_train = train_npz['X']
    y_train = train_npz['y']
    test_npz = np.load(out / 'test_preprocessed.npz')
    X_test = test_npz['X']

    pos = int((y_train == 1).sum())
    neg = len(y_train) - pos
    spw = neg / max(pos, 1)

    # base predictions: lgb/xgb from preprocessed; te/cat from raw features
    preds = {}

    if 'lgb' in names:
        lgb_params = {
            'objective': 'binary',
            'metric': 'None',
            'verbose': -1,
            'scale_pos_weight': spw,
        }
        lgb_model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), num_boost_round=100)
        preds['lgb'] = lgb_model.predict(X_test)

    if 'xgb' in names:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            scale_pos_weight=spw,
        )
        xgb_model.fit(X_train, y_train)
        preds['xgb'] = xgb_model.predict_proba(X_test)[:, 1]

    # TE and Cat use raw with add_features
    if 'te' in names or 'cat' in names:
        train_df = pd.read_csv(data_dir / 'train.csv')
        test_df = pd.read_csv(data_dir / 'test.csv')
        id_col = 'id'
        target_col = 'LoanStatus'
        Xtr = add_features(train_df.drop(columns=[target_col]))
        ytr = train_df[target_col].values
        Xts = add_features(test_df.copy())
        cat_cols = Xtr.select_dtypes(include='object').columns.tolist()
        Xtr_noid = Xtr.drop(columns=[id_col])
        Xts_noid = Xts.drop(columns=[id_col])

        if 'te' in names:
            te = TargetEncoder(cols=cat_cols)
            Xtr_enc = te.fit_transform(Xtr_noid, ytr)
            Xts_enc = te.transform(Xts_noid)

            # try to reuse optuna params if exist
            params_path = out / 'optuna_best_params.json'
            if params_path.exists():
                with params_path.open('r', encoding='utf-8') as f:
                    best_params = json.load(f).get('best_params', {})
            else:
                best_params = {}
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                objective='binary',
                random_state=42,
                n_estimators=200,
                learning_rate=best_params.get('learning_rate', 0.1),
                num_leaves=best_params.get('num_leaves', 31),
                max_depth=best_params.get('max_depth', -1),
                min_child_samples=best_params.get('min_data_in_leaf', 20),
                feature_fraction=best_params.get('feature_fraction', 1.0),
                bagging_fraction=best_params.get('bagging_fraction', 1.0),
                lambda_l1=best_params.get('lambda_l1', 0.0),
                lambda_l2=best_params.get('lambda_l2', 0.0),
                scale_pos_weight=best_params.get('scale_pos_weight', spw),
            )
            model.fit(Xtr_enc, ytr)
            preds['te'] = model.predict_proba(Xts_enc)[:, 1]

        if 'cat' in names:
            model = CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=False, random_state=42
            )
            model.fit(Xtr_noid, ytr, cat_features=cat_cols)
            preds['cat'] = model.predict_proba(Xts_noid)[:, 1]

    # ensure order matches coeffs
    X_meta_test = np.vstack([preds[name] for name in names]).T
    logits = X_meta_test @ coef + intercept
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob > thr).astype(int)

    test_ids = pd.read_csv(data_dir / 'test.csv')['id']
    submission = pd.DataFrame({'Id': test_ids, 'LoanStatus': pred})
    submission.to_csv(out / 'submit_stack.csv', index=False, header=False)


if __name__ == '__main__':
    main()

