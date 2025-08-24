import json
from pathlib import Path

import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "data" / "input" / "train.csv"
    output_dir = base_dir / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_by_column": df.isnull().sum().to_dict(),
        "target_distribution": df["LoanStatus"].value_counts(normalize=True).to_dict(),
        "feature_names": [c for c in df.columns if c != "LoanStatus"],
    }

    output_file = output_dir / "eda_summary.json"
    with output_file.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"EDA summary saved to {output_file}")


if __name__ == "__main__":
    main()
