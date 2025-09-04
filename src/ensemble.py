import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
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
    lgb_oof = np.zeros(len(y))
    xgb_oof = np.zeros(len(y))

    for train_idx, valid_idx in cv.split(X, y):
        # imbalance handling
        pos = np.sum(y[train_idx] == 1)
        neg = len(train_idx) - pos
        spw = neg / max(pos, 1)
        lgb_train = lgb.Dataset(X[train_idx], label=y[train_idx])
        lgb_valid = lgb.Dataset(X[valid_idx], label=y[valid_idx])
        lgb_params = {
            "objective": "binary",
            "metric": "None",
            "learning_rate": 0.1,
            "verbose": -1,
            "scale_pos_weight": spw,
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
            scale_pos_weight=spw,
        )
        xgb_model.fit(X[train_idx], y[train_idx])
        xgb_oof[valid_idx] = xgb_model.predict_proba(X[valid_idx])[:, 1]

    # Optionally include TE and Cat OOF if present (must align by index)
    oof_dict = {"lgb": lgb_oof, "xgb": xgb_oof}
    for name, path in ("te", output_dir / "oof_te.npy"), ("cat", output_dir / "oof_cat.npy"):
        if path.exists():
            try:
                oof_dict[name] = np.load(path)
            except Exception:
                pass

    # score single-model performance and select top K to avoid weak models degrading blend
    single_scores = {}
    thresholds = np.linspace(0.0, 1.0, 101)
    for name, arr in oof_dict.items():
        best = 0.0
        for thr in thresholds:
            f1 = f1_score(y, (arr > thr).astype(int))
            if f1 > best:
                best = f1
        single_scores[name] = float(best)
    # pick top 3 models by single-model F1 (or all if <=3)
    sorted_names = sorted(single_scores.keys(), key=lambda n: single_scores[n], reverse=True)
    model_names = sorted_names[:3]
    oofs = [oof_dict[k] for k in model_names]

    thresholds = np.linspace(0.0, 1.0, 101)

    def evaluate_blend(weights: np.ndarray) -> tuple[float, float]:
        blended = np.zeros(len(y))
        for w, arr in zip(weights, oofs):
            blended += w * arr
        best_f1 = -1.0
        best_thr = 0.5
        for thr in thresholds:
            f1 = f1_score(y, (blended > thr).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        return float(best_f1), float(best_thr)

    best = {"f1": -1.0, "weights": None, "threshold": 0.5}
    step = 0.05
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    if len(oofs) == 2:
        for w in grid:
            weights = np.array([w, 1 - w])
            f1, thr = evaluate_blend(weights)
            if f1 > best["f1"]:
                best = {"f1": f1, "weights": weights.tolist(), "threshold": thr}
    elif len(oofs) == 3:
        for w1 in grid:
            for w2 in grid:
                w3 = 1 - w1 - w2
                if w3 < -1e-9:
                    continue
                weights = np.array([w1, w2, max(0.0, w3)])
                # renormalize to sum=1
                s = weights.sum()
                if s == 0:
                    continue
                weights = weights / s
                f1, thr = evaluate_blend(weights)
                if f1 > best["f1"]:
                    best = {"f1": f1, "weights": weights.tolist(), "threshold": thr}
    else:
        # fallback for unexpected counts
        w = np.ones(len(oofs)) / len(oofs)
        f1, thr = evaluate_blend(w)
        best = {"f1": f1, "weights": w.tolist(), "threshold": thr}

    results = {
        "models": model_names,
        "weights": best["weights"],
        "best_threshold": best["threshold"],
        "best_f1": best["f1"],
    }
    with open(output_dir / "ensemble_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    np.savez_compressed(output_dir / "ensemble_oof_preds.npz", **{k: v for k, v in oof_dict.items()})


if __name__ == "__main__":
    main()
