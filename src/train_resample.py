import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def evaluate_with_sampler(sampler, X, y, params, cv):
    scores = []
    for train_idx, valid_idx in cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[valid_data],
            callbacks=[lgb.log_evaluation(0)],
        )
        preds = (model.predict(X_valid) > 0.5).astype(int)
        scores.append(f1_score(y_valid, preds))
    return scores


def main():
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "data" / "output"
    data = np.load(output_dir / "train_preprocessed.npz")
    X = data["X"]
    y = data["y"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    params = {"objective": "binary", "metric": "None", "learning_rate": 0.1, "verbose": -1}
    samplers = {
        "none": None,
        "smote": SMOTE(random_state=42),
        "under": RandomUnderSampler(random_state=42),
    }
    results = {}
    for name, sampler in samplers.items():
        scores = evaluate_with_sampler(sampler, X, y, params, cv)
        results[name] = {"f1_scores": [float(s) for s in scores], "mean_f1": float(np.mean(scores))}
    with open(output_dir / "resample_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
