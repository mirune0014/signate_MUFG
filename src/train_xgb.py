import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def main():
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "data" / "output"
    data = np.load(output_dir / "train_preprocessed.npz")
    X = data["X"]
    y = data["y"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    oof = np.zeros(len(y), dtype=float)
    # try to load tuned params
    params_path = output_dir / 'optuna_xgb_best.json'
    tuned = {}
    if params_path.exists():
        with params_path.open('r', encoding='utf-8') as f:
            tuned = json.load(f).get('best_params', {})

    for train_idx, valid_idx in cv.split(X, y):
        pos = np.sum(y[train_idx] == 1)
        neg = len(train_idx) - pos
        spw = neg / max(pos, 1)
        base_params = dict(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            scale_pos_weight=spw,
        )
        base_params.update(tuned)
        model = xgb.XGBClassifier(**base_params)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[valid_idx])[:, 1]
        oof[valid_idx] = proba
        preds = (proba > 0.5).astype(int)
        scores.append(f1_score(y[valid_idx], preds))

    results = {"f1_scores": scores, "mean_f1": float(np.mean(scores))}
    with open(output_dir / "xgb_cv_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    np.save(output_dir / 'oof_xgb.npy', oof)


if __name__ == "__main__":
    main()
