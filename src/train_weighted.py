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

    pos = np.sum(y == 1)
    neg = len(y) - pos
    base_weight = neg / pos
    weights = [base_weight * 0.8, base_weight, base_weight * 1.2]

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for w in weights:
        scores = []
        for train_idx, valid_idx in cv.split(X, y):
            train_data = lgb.Dataset(X[train_idx], label=y[train_idx])
            valid_data = lgb.Dataset(X[valid_idx], label=y[valid_idx])
            params = {
                "objective": "binary",
                "metric": "None",
                "learning_rate": 0.1,
                "verbose": -1,
                "scale_pos_weight": w,
            }
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=[lgb.log_evaluation(0)],
            )
            preds = (model.predict(X[valid_idx]) > 0.5).astype(int)
            scores.append(f1_score(y[valid_idx], preds))
        results[w] = scores

    summary = {
        str(w): {"f1_scores": s, "mean_f1": float(np.mean(s))} for w, s in results.items()
    }
    with open(output_dir / "class_weight_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
