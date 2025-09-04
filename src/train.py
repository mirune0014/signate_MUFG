import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def main():
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "data" / "output"
    data = np.load(output_dir / "train_preprocessed.npz")
    X = data["X"]
    y = data["y"]

    pos = np.sum(y == 1)
    neg = len(y) - pos
    scale_pos_weight = neg / pos

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    oof_preds = np.zeros(len(y))
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
        train_data = lgb.Dataset(X[train_idx], label=y[train_idx])
        valid_data = lgb.Dataset(X[valid_idx], label=y[valid_idx])
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "verbose": -1,
            "scale_pos_weight": scale_pos_weight,
            "num_leaves": 63,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 0,
        }
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=2000,
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        oof = model.predict(X[valid_idx], num_iteration=model.best_iteration)
        oof_preds[valid_idx] = oof
        preds = (oof > 0.5).astype(int)
        score = f1_score(y[valid_idx], preds)
        scores.append(score)

    np.save(output_dir / "oof_preds.npy", oof_preds)
    results = {"f1_scores": scores, "mean_f1": float(np.mean(scores))}
    with open(output_dir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
