import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def main():
    output_dir = Path(r"C:\Users\miots\ruruprojects3\MUFG\signate_MUFG\data\output")
    data = np.load(output_dir / "train_preprocessed.npz")
    X = data["X"]
    y = data["y"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof = np.zeros(len(y))
    xgb_oof = np.zeros(len(y))

    for train_idx, valid_idx in cv.split(X, y):
        lgb_train = lgb.Dataset(X[train_idx], label=y[train_idx])
        lgb_valid = lgb.Dataset(X[valid_idx], label=y[valid_idx])
        lgb_params = {
            "objective": "binary",
            "metric": "None",
            "learning_rate": 0.1,
            "verbose": -1,
        }
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_valid],
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)],
        )
        lgb_oof[valid_idx] = lgb_model.predict(X[valid_idx])

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
        xgb_model.fit(X[train_idx], y[train_idx])
        xgb_oof[valid_idx] = xgb_model.predict_proba(X[valid_idx])[:, 1]

    avg_oof = (lgb_oof + xgb_oof) / 2
    threshold = 0.28
    ensemble_f1 = f1_score(y, (avg_oof > threshold).astype(int))
    results = {"ensemble_f1": ensemble_f1}
    with open(output_dir / "ensemble_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    np.savez_compressed(output_dir / "ensemble_oof_preds.npz", lgb=lgb_oof, xgb=xgb_oof, ensemble=avg_oof)


if __name__ == "__main__":
    main()
