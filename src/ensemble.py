import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def search_best_threshold(y_true, probs):
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0, 1, 101):
        f1 = f1_score(y_true, (probs >= thr).astype(int))
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return best_thr, best_f1


def main():
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "data" / "output"

    data = np.load(output_dir / "train_preprocessed.npz")
    X = data["X"]
    y = data["y"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof = np.zeros(len(y))
    xgb_oof = np.zeros(len(y))
    cat_oof = np.zeros(len(y))

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

        cat_model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=3.0,
            verbose=False,
            random_state=42,
        )
        cat_model.fit(X[train_idx], y[train_idx])
        cat_oof[valid_idx] = cat_model.predict_proba(X[valid_idx])[:, 1]

    lgb_thr, lgb_f1 = search_best_threshold(y, lgb_oof)
    xgb_thr, xgb_f1 = search_best_threshold(y, xgb_oof)
    cat_thr, cat_f1 = search_best_threshold(y, cat_oof)

    simple_oof = (lgb_oof + xgb_oof + cat_oof) / 3
    simple_thr, simple_f1 = search_best_threshold(y, simple_oof)

    weights = np.array([lgb_f1, xgb_f1, cat_f1])
    weighted_oof = (lgb_oof * weights[0] + xgb_oof * weights[1] + cat_oof * weights[2]) / weights.sum()
    weighted_thr, weighted_f1 = search_best_threshold(y, weighted_oof)

    meta_features = np.column_stack([lgb_oof, xgb_oof, cat_oof])
    stack_oof = np.zeros(len(y))
    for train_idx, valid_idx in cv.split(meta_features, y):
        meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(meta_features[train_idx], y[train_idx])
        stack_oof[valid_idx] = meta_model.predict_proba(meta_features[valid_idx])[:, 1]
    stack_thr, stack_f1 = search_best_threshold(y, stack_oof)

    results = {
        "lgb_f1": lgb_f1,
        "lgb_threshold": lgb_thr,
        "xgb_f1": xgb_f1,
        "xgb_threshold": xgb_thr,
        "cat_f1": cat_f1,
        "cat_threshold": cat_thr,
        "simple_avg_f1": simple_f1,
        "simple_avg_threshold": simple_thr,
        "weighted_avg_f1": weighted_f1,
        "weighted_avg_threshold": weighted_thr,
        "stacking_f1": stack_f1,
        "stacking_threshold": stack_thr,
        "weighted_weights": weights.tolist(),
    }
    with open(output_dir / "ensemble_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    np.savez_compressed(
        output_dir / "ensemble_oof_preds.npz",
        lgb=lgb_oof,
        xgb=xgb_oof,
        cat=cat_oof,
        simple=simple_oof,
        weighted=weighted_oof,
        stacking=stack_oof,
    )


if __name__ == "__main__":
    main()
