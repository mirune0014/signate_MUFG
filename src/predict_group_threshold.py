import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    data_dir = base / 'data' / 'input'
    out = base / 'data' / 'output'

    cfg = json.loads((out / 'threshold_group_results.json').read_text(encoding='utf-8'))
    group = cfg.get('group')
    thr_map = cfg.get('thresholds', {})
    global_thr = float(cfg.get('global_threshold', 0.5))

    prob = np.load(out / 'test_blend_prob.npy')
    test = pd.read_csv(data_dir / 'test.csv')
    if group is None:
        pred = (prob > global_thr).astype(int)
    else:
        keys = test[group].astype(str).values
        pred = np.zeros_like(prob, dtype=int)
        for i, p in enumerate(prob):
            t = float(thr_map.get(str(keys[i]), global_thr))
            pred[i] = 1 if p > t else 0

    sub = pd.DataFrame({'Id': test['id'], 'LoanStatus': pred})
    sub.to_csv(out / 'submit_te_xgb_group.csv', index=False, header=False)


if __name__ == '__main__':
    main()

