import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def main():
    output_dir = Path(r"C:\Users\miots\ruruprojects3\MUFG\signate_MUFG\data\output")
    data = np.load(output_dir / "train_preprocessed.npz")
    X = data["X"]
    y = data["y"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in cv.split(X, y):
        model = xgb.XGBClassifier(
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
        model.fit(X[train_idx], y[train_idx])
        preds = (model.predict_proba(X[valid_idx])[:, 1] > 0.5).astype(int)
        scores.append(f1_score(y[valid_idx], preds))

    results = {"f1_scores": scores, "mean_f1": float(np.mean(scores))}
    with open(output_dir / "xgb_cv_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
