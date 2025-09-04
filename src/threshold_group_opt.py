import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


GROUP_CANDIDATES = [
    'Subprogram', 'NaicsSector', 'ApprovalFiscalYear', 'BusinessType'
]


def main(min_support: int = 200) -> None:
    base = Path(__file__).resolve().parents[1]
    data_dir = base / 'data' / 'input'
    out = base / 'data' / 'output'

    y = np.load(out / 'train_preprocessed.npz')['y']
    # load OOFs
    oofs = {}
    if (out / 'oof_te.npy').exists():
        oofs['te'] = np.load(out / 'oof_te.npy')
    if (out / 'oof_xgb.npy').exists():
        oofs['xgb'] = np.load(out / 'oof_xgb.npy')
    if (out / 'oof_cat.npy').exists():
        oofs['cat'] = np.load(out / 'oof_cat.npy')
    if not oofs:
        raise RuntimeError('No OOF predictions found.')

    cfg = json.loads((out / 'ensemble_te_xgb.json').read_text(encoding='utf-8'))
    if 'weights' in cfg:
        names = cfg.get('models', ['te','xgb','cat'])
        ws = np.array(cfg['weights'], dtype=float)
        ws = ws / max(ws.sum(), 1e-9)
        blend_oof = np.zeros_like(next(iter(oofs.values())))
        for w, n in zip(ws, names):
            if n in oofs:
                blend_oof += w * oofs[n]
    else:
        w_te = float(cfg.get('weight_te', 0.5))
        blend_oof = w_te * oofs.get('te', 0.0) + (1 - w_te) * oofs.get('xgb', 0.0)

    # compute global best
    thresholds = np.linspace(0.0, 1.0, 101)
    scores = [f1_score(y, (blend_oof > t).astype(int)) for t in thresholds]
    g_idx = int(np.argmax(scores))
    global_thr = float(thresholds[g_idx])
    global_f1 = float(scores[g_idx])

    # load raw train for grouping keys
    train_df = pd.read_csv(data_dir / 'train.csv')
    best = {
        'group': None,
        'thresholds': {},
        'min_support': min_support,
        'global_threshold': global_thr,
        'global_f1': global_f1,
        'group_f1': global_f1,
    }
    for col in GROUP_CANDIDATES:
        keys = train_df[col].values
        # thresholds per group
        thr_map = {}
        preds = np.zeros_like(y, dtype=int)
        for k in np.unique(keys):
            idx = np.where(keys == k)[0]
            if len(idx) < min_support:
                thr_map[k] = global_thr
                preds[idx] = (blend_oof[idx] > global_thr).astype(int)
                continue
            scs = [f1_score(y[idx], (blend_oof[idx] > t).astype(int)) for t in thresholds]
            ti = int(np.argmax(scs))
            tbest = float(thresholds[ti])
            thr_map[k] = tbest
            preds[idx] = (blend_oof[idx] > tbest).astype(int)
        f1 = float(f1_score(y, preds))
        if f1 > best['group_f1']:
            best = {
                'group': col,
                'thresholds': {str(k): float(v) for k, v in thr_map.items()},
                'min_support': min_support,
                'global_threshold': global_thr,
                'global_f1': global_f1,
                'group_f1': f1,
            }

    out.write_text if False else None
    with (out / 'threshold_group_results.json').open('w', encoding='utf-8') as f:
        json.dump(best, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--min-support', type=int, default=200)
    args = p.parse_args()
    main(min_support=args.min_support)

