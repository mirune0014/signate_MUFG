"""Train XGB with pseudo-labeled test (high-confidence) and create submission.

Outputs: data/output/submit_xgb_pseudo.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    out = base / 'data' / 'output'
    inp = base / 'data' / 'input'

    # load preprocessed matrices
    train_npz = np.load(out / 'train_preprocessed.npz')
    X = train_npz['X']
    y = train_npz['y']
    X_test = np.load(out / 'test_preprocessed.npz')['X']

    # tuned params if any
    tuned = {}
    xgb_cfg = out / 'optuna_xgb_best.json'
    if xgb_cfg.exists():
        with xgb_cfg.open('r', encoding='utf-8') as f:
            tuned = json.load(f).get('best_params', {})
    pos = int((y == 1).sum()); neg = len(y)-pos
    base_params = dict(
        n_estimators=max(400, int(tuned.get('n_estimators', 400))),
        learning_rate=tuned.get('learning_rate', 0.05),
        max_depth=tuned.get('max_depth', 6),
        min_child_weight=tuned.get('min_child_weight', 1.0),
        subsample=tuned.get('subsample', 0.8),
        colsample_bytree=tuned.get('colsample_bytree', 0.8),
        gamma=tuned.get('gamma', 0.0),
        reg_alpha=tuned.get('reg_alpha', 0.0),
        reg_lambda=tuned.get('reg_lambda', 1.0),
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        scale_pos_weight=tuned.get('scale_pos_weight', neg/max(pos,1)),
    )

    # 1) fit on full train, get test probas
    model = xgb.XGBClassifier(**base_params)
    model.fit(X, y)
    proba_test = model.predict_proba(X_test)[:, 1]

    # 2) select high-confidence pseudo labels
    hi_pos = proba_test >= 0.98
    hi_neg = proba_test <= 0.02
    mask = hi_pos | hi_neg
    if mask.sum() == 0:
        # fallback to create standard submission
        thr = 0.5
        sub = pd.DataFrame({'Id': pd.read_csv(inp / 'test.csv')['id'], 'LoanStatus': (proba_test>thr).astype(int)})
        sub.to_csv(out / 'submit_xgb_pseudo.csv', index=False, header=False)
        return

    X_aug = np.vstack([X, X_test[mask]])
    y_aug = np.concatenate([y, (proba_test[mask] >= 0.5).astype(int)])

    # 3) refit with augmented data (slightly stronger model)
    params2 = dict(base_params)
    params2['n_estimators'] = int(base_params['n_estimators'] * 1.2)
    model2 = xgb.XGBClassifier(**params2)
    model2.fit(X_aug, y_aug)
    prob2 = model2.predict_proba(X_test)[:, 1]

    # threshold: prefer ensemble/optuna threshold if exists
    thr = 0.5
    thr_paths = ['optuna_xgb_best.json', 'ensemble_results.json', 'ensemble_te_xgb.json']
    for p in thr_paths:
        f = out / p
        if f.exists():
            try:
                d = json.loads(f.read_text(encoding='utf-8'))
                thr = float(d.get('best_threshold', thr))
                break
            except Exception:
                pass
    pred = (prob2 > thr).astype(int)
    sub = pd.DataFrame({'Id': pd.read_csv(inp / 'test.csv')['id'], 'LoanStatus': pred})
    sub.to_csv(out / 'submit_xgb_pseudo.csv', index=False, header=False)


if __name__ == '__main__':
    main()

