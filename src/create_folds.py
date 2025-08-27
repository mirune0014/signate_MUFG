import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "input"
    output_dir = base_dir / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "train.csv")
    target_col = "LoanStatus"
    groups = df["ApprovalFiscalYear"]

    gkf = GroupKFold(n_splits=5)
    folds = []
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(df, df[target_col], groups), 1):
        folds.append(
            {
                "train_idx": train_idx.tolist(),
                "valid_idx": valid_idx.tolist(),
                "group_val": groups.iloc[valid_idx].unique().tolist(),
            }
        )

    with open(output_dir / "folds.json", "w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
