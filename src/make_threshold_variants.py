from pathlib import Path
import numpy as np
import pandas as pd


def main():
    base = Path(__file__).resolve().parents[1]
    out = base / 'data' / 'output'
    data = base / 'data' / 'input'

    prob_path = out / 'test_blend_prob.npy'
    if not prob_path.exists():
        raise SystemExit('Run predict_te_xgb_blend.py first to create test_blend_prob.npy')
    prob = np.load(prob_path)
    ids = pd.read_csv(data / 'test.csv')['id']

    # try a small grid around 0.40
    thrs = [0.36, 0.38, 0.39, 0.40, 0.41, 0.42, 0.44]
    for t in thrs:
        pred = (prob > t).astype(int)
        sub = pd.DataFrame({'Id': ids, 'LoanStatus': pred})
        sub.to_csv(out / f'submit_te_xgb_thr{int(t*100):02d}.csv', index=False, header=False)


if __name__ == '__main__':
    main()

