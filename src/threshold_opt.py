import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def main():

    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "data" / "output"

    data = np.load(output_dir / "train_preprocessed.npz")
    y = data["y"]
    oof_preds = np.load(output_dir / "oof_preds.npy")

    thresholds = np.linspace(0, 1, 101)
    scores = []
    for thr in thresholds:
        preds = (oof_preds > thr).astype(int)
        scores.append(f1_score(y, preds))

    best_idx = int(np.argmax(scores))
    results = {
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(scores[best_idx]),
        "thresholds": thresholds.tolist(),
        "f1_scores": scores,
    }
    with open(output_dir / "threshold_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
