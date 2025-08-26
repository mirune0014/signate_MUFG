import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def main():

    output_dir = Path(r"C:\Users\miots\ruruprojects3\MUFG\signate_MUFG\data\output")

    data = np.load(output_dir / "train_preprocessed.npz")
    X = data["X"]
    y = data["y"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    for train_idx, valid_idx in cv.split(X, y):
        train_data = lgb.Dataset(X[train_idx], label=y[train_idx])
        valid_data = lgb.Dataset(X[valid_idx], label=y[valid_idx])
        params = {
            "objective": "binary",
            "metric": "None",
            "learning_rate": 0.1,
            "verbose": -1,
        }
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)],
        )
        oof_preds[valid_idx] = model.predict(X[valid_idx])

    thresholds = np.linspace(0, 1, 101)
    scores = []
    for thr in thresholds:
        preds = (oof_preds > thr).astype(int)
        scores.append(f1_score(y, preds))

    best_idx = int(np.argmax(scores))
    results = {
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(scores[best_idx]),
        "thresholds": thresholds.tolist(),
        "f1_scores": scores,
    }
    with open(output_dir / "threshold_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
