import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    out = base_dir / 'data' / 'output'
    y = np.load(out / 'train_preprocessed.npz')['y']
    oof_te = np.load(out / 'oof_te.npy')
    oof_xgb = np.load(out / 'oof_xgb.npy')
    oofs = [oof_te, oof_xgb]
    names = ['te', 'xgb']
    # optionally include CatBoost if available
    cat_path = out / 'oof_cat.npy'
    if cat_path.exists():
        try:
            oof_cat = np.load(cat_path)
            oofs.append(oof_cat)
            names.append('cat')
        except Exception:
            pass

    thresholds = np.linspace(0.0, 1.0, 101)
    if len(oofs) == 2:
        weights = np.linspace(0.0, 1.0, 101)
        best = {'weight_te': 0.5, 'threshold': 0.5, 'f1': -1.0}
        for w in weights:
            blended = w * oofs[0] + (1 - w) * oofs[1]
            for thr in thresholds:
                f1 = f1_score(y, (blended > thr).astype(int))
                if f1 > best['f1']:
                    best = {'weight_te': float(w), 'threshold': float(thr), 'f1': float(f1)}
    else:
        # grid search over 3 weights that sum to 1
        grid = np.arange(0.0, 1.0 + 1e-9, 0.05)
        best = {'weights': [1/3, 1/3, 1/3], 'threshold': 0.5, 'f1': -1.0, 'models': names}
        for w1 in grid:
            for w2 in grid:
                w3 = 1 - w1 - w2
                if w3 < -1e-9:
                    continue
                weights = np.array([w1, w2, max(0.0, w3)])
                s = weights.sum()
                if s == 0:
                    continue
                weights = weights / s
                blended = weights[0] * oofs[0] + weights[1] * oofs[1] + weights[2] * oofs[2]
                for thr in thresholds:
                    f1 = f1_score(y, (blended > thr).astype(int))
                    if f1 > best['f1']:
                        best = {'weights': weights.tolist(), 'threshold': float(thr), 'f1': float(f1), 'models': names}

    out.mkdir(parents=True, exist_ok=True)
    with (out / 'ensemble_te_xgb.json').open('w', encoding='utf-8') as f:
        json.dump(best, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
