import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    out = base_dir / 'data' / 'output'
    data = np.load(out / 'train_preprocessed.npz')
    y = data['y']

    # load available OOF predictions
    feats = []
    names = []
    paths = {
        'lgb': out / 'oof_preds.npy',
        'xgb': out / 'oof_xgb.npy',
        'te': out / 'oof_te.npy',
        'cat': out / 'oof_cat.npy',
    }
    for name, p in paths.items():
        if p.exists():
            feats.append(np.load(p))
            names.append(name)

    if len(feats) < 2:
        raise RuntimeError('Need at least two OOF prediction sources to stack')

    X_meta = np.vstack(feats).T  # shape (n_samples, n_models)

    # Meta model with CV for OOF meta predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_meta = np.zeros(len(y))
    for tr, va in cv.split(X_meta, y):
        clf = LogisticRegression(max_iter=1000, n_jobs=None, class_weight='balanced', solver='lbfgs')
        clf.fit(X_meta[tr], y[tr])
        oof_meta[va] = clf.predict_proba(X_meta[va])[:, 1]

    # threshold grid
    thresholds = np.linspace(0.0, 1.0, 101)
    scores = [f1_score(y, (oof_meta > t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(scores))
    best_thr = float(thresholds[best_idx])
    best_f1 = float(scores[best_idx])

    # fit final meta on all data and persist coefficients
    meta = LogisticRegression(max_iter=1000, n_jobs=None, class_weight='balanced', solver='lbfgs')
    meta.fit(X_meta, y)
    coefs = meta.coef_.ravel().tolist()
    intercept = float(meta.intercept_[0])

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'stacking_oof.npy', oof_meta)
    with (out / 'stacking_results.json').open('w', encoding='utf-8') as f:
        json.dump(
            {
                'models': names,
                'best_threshold': best_thr,
                'best_f1': best_f1,
                'coef': coefs,
                'intercept': intercept,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == '__main__':
    main()

