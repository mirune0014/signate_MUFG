import json
from pathlib import Path

import numpy as np
import optuna
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

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "None",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "bagging_freq": 0,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", base_weight * 0.5, base_weight * 1.5),
            "verbose": -1,
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, valid_idx in cv.split(X, y):
            train_data = lgb.Dataset(X[train_idx], label=y[train_idx])
            valid_data = lgb.Dataset(X[valid_idx], label=y[valid_idx])
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=[lgb.log_evaluation(0)],
            )
            preds = (model.predict(X[valid_idx]) > 0.5).astype(int)
            scores.append(f1_score(y[valid_idx], preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    best = {
        "best_params": study.best_params,
        "best_score": study.best_value,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
