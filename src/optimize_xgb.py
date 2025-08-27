import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def main(trials: int) -> None:
    output_dir = Path(__file__).resolve().parents[1] / "data" / "output"
    data = np.load(output_dir / "train_preprocessed.npz")
    X = data["X"]
    y = data["y"]

    pos = np.sum(y == 1)
    neg = len(y) - pos
    base_weight = neg / pos

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", base_weight * 0.5, base_weight * 1.5),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "tree_method": "hist",
            "eval_metric": "logloss",
            "n_jobs": -1,
            "verbosity": 0,
            "random_state": 42,
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, valid_idx in cv.split(X, y):
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_idx], y[train_idx])
            preds = (model.predict_proba(X[valid_idx])[:, 1] > 0.5).astype(int)
            scores.append(f1_score(y[valid_idx], preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)

    best = {"best_params": study.best_params, "best_score": study.best_value}
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "xgb_optuna_results.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    main(args.trials)
