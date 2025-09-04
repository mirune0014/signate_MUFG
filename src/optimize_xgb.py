import json
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def main(n_trials: int = 50) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    out = base_dir / 'data' / 'output'
    data = np.load(out / 'train_preprocessed.npz')
    X = data['X']
    y = data['y']

    pos = int((y == 1).sum())
    neg = len(y) - pos
    base_spw = neg / max(pos, 1)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', base_spw * 0.5, base_spw * 1.5),
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0,
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        for tr, va in cv.split(X, y):
            model = xgb.XGBClassifier(**params)
            model.fit(X[tr], y[tr])
            oof[va] = model.predict_proba(X[va])[:, 1]
        thresholds = np.linspace(0.0, 1.0, 101)
        scores = [f1_score(y, (oof > t).astype(int)) for t in thresholds]
        return float(np.max(scores))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # After best trial, re-evaluate to get best threshold
    best_params = study.best_params
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        model = xgb.XGBClassifier(**best_params, eval_metric='logloss', n_jobs=-1, random_state=42, verbosity=0)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
    thresholds = np.linspace(0.0, 1.0, 101)
    scores = [f1_score(y, (oof > t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(scores))
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'optuna_xgb_best.json').open('w', encoding='utf-8') as f:
        json.dump({'best_params': best_params, 'best_score': float(np.max(scores)), 'best_threshold': float(thresholds[best_idx])}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=50)
    args = p.parse_args()
    main(n_trials=args.trials)
