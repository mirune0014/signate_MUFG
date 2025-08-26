import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb


def main():
    data_dir = Path(r"C:\Users\miots\ruruprojects3\MUFG\signate_MUFG\data\input")
    output_dir = Path(r"C:\Users\miots\ruruprojects3\MUFG\signate_MUFG\data\output")
    train_data = np.load(output_dir / "train_preprocessed.npz")
    X_train = train_data["X"]
    y_train = train_data["y"]
    test_data = np.load(output_dir / "test_preprocessed.npz")
    X_test = test_data["X"]

    params_path = output_dir / "optuna_best_params.json"
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)["best_params"]
    else:
        best_params = {"learning_rate": 0.1, "num_leaves": 31}

    lgb_params = {"objective": "binary", "metric": "None", "verbose": -1, **best_params}
    lgb_model = lgb.train(
        lgb_params,
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=100,
    )
    lgb_pred = lgb_model.predict(X_test)

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
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]

    pred_prob = (lgb_pred + xgb_pred) / 2
    pred = (pred_prob > 0.5).astype(int)

    test_ids = pd.read_csv(data_dir / "test.csv")["id"]
    submission = pd.DataFrame({"Id": test_ids, "LoanStatus": pred})
    submission.to_csv(output_dir / "submit4.csv", index=False, header=False)


if __name__ == "__main__":
    main()
