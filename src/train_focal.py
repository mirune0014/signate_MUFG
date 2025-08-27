import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def focal_loss_lgb(y_pred, dtrain, alpha=0.25, gamma=2.0):
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    grad = (p - y_true)
    grad *= (alpha * y_true + (1 - alpha) * (1 - y_true))
    grad *= ((1 - p) ** gamma * y_true + p ** gamma * (1 - y_true))
    hess = (
        (alpha * y_true + (1 - alpha) * (1 - y_true))
        * ((1 - p) ** gamma * y_true + p ** gamma * (1 - y_true))
        * (
            p
            * (1 - p)
            * (
                gamma
                * (
                    (y_true - p)
                    * ((y_true - 1) * p - y_true * (p - 1))
                    / (p * (1 - p))
                )
                + 1
            )
        )
    )
    return grad, hess


def main():
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "data" / "output"
    data = np.load(output_dir / "train_preprocessed.npz")
    X = data["X"]
    y = data["y"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    params = {"objective": focal_loss_lgb, "metric": "None", "learning_rate": 0.1, "verbose": -1}
    scores = []
    for train_idx, valid_idx in cv.split(X, y):
        train_data = lgb.Dataset(X[train_idx], label=y[train_idx])
        valid_data = lgb.Dataset(X[valid_idx], label=y[valid_idx])
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[valid_data],
            callbacks=[lgb.log_evaluation(0)],
        )
        preds = (model.predict(X[valid_idx]) > 0.5).astype(int)
        scores.append(f1_score(y[valid_idx], preds))
    results = {"f1_scores": [float(s) for s in scores], "mean_f1": float(np.mean(scores))}
    with open(output_dir / "focal_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
