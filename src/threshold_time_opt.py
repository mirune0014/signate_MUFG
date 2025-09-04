import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    out = base / 'data' / 'output'
    data_dir = base / 'data' / 'input'

    y = np.load(out / 'train_preprocessed.npz')['y']
    train_df = pd.read_csv(data_dir / 'train.csv')
    years = train_df['ApprovalFiscalYear'].values

    # reconstruct blended OOF based on ensemble config
    oofs = {}
    if (out / 'oof_te.npy').exists():
        oofs['te'] = np.load(out / 'oof_te.npy')
    if (out / 'oof_xgb.npy').exists():
        oofs['xgb'] = np.load(out / 'oof_xgb.npy')
    if (out / 'oof_cat.npy').exists():
        oofs['cat'] = np.load(out / 'oof_cat.npy')
    cfg = json.loads((out / 'ensemble_te_xgb.json').read_text(encoding='utf-8'))
    if 'weights' in cfg:
        names = cfg.get('models', ['te','xgb','cat'])
        ws = np.array(cfg['weights'], dtype=float)
        ws = ws / max(ws.sum(), 1e-9)
        blend = np.zeros_like(next(iter(oofs.values())))
        for w, n in zip(ws, names):
            if n in oofs:
                blend += w * oofs[n]
    else:
        w_te = float(cfg.get('weight_te', 0.5))
        blend = w_te * oofs.get('te', 0.0) + (1 - w_te) * oofs.get('xgb', 0.0)

    # choose threshold on the most-recent ~30% of years (by unique-year count)
    uq = np.unique(years)
    k_recent = max(1, int(np.ceil(len(uq) * 0.3)))
    recent_years = set(uq[-k_recent:])
    idx_recent = np.where(np.isin(years, list(recent_years)))[0]
    idx_rest = np.where(~np.isin(years, list(recent_years)))[0]

    thresholds = np.linspace(0.0, 1.0, 101)
    # pick threshold maximizing recent F1
    scores_recent = [f1_score(y[idx_recent], (blend[idx_recent] > t).astype(int)) for t in thresholds]
    bi = int(np.argmax(scores_recent))
    best_thr = float(thresholds[bi])

    # report overall/segment scores for info
    f1_recent = float(scores_recent[bi])
    f1_rest = float(f1_score(y[idx_rest], (blend[idx_rest] > best_thr).astype(int))) if len(idx_rest) else f1_recent
    f1_all = float(f1_score(y, (blend > best_thr).astype(int)))

    res = {
        'best_threshold': best_thr,
        'segment': 'recent_years',
        'recent_years': sorted(list(map(int, recent_years))),
        'k_recent_unique_years': k_recent,
        'f1_recent': f1_recent,
        'f1_rest': f1_rest,
        'f1_all': f1_all,
    }
    with (out / 'threshold_time_results.json').open('w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

