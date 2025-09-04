"""Batch runner for experiments: tuning + OOF + ensembles + submissions.

Usage examples:
  python src/run_experiments.py --xgb-trials 150 --cat-trials 120 --te-trials 80 --do-stacking
"""

from pathlib import Path
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    src = base / 'src'

    # parse args (basic)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--xgb-trials', type=int, default=120)
    p.add_argument('--cat-trials', type=int, default=120)
    p.add_argument('--te-trials', type=int, default=80)
    p.add_argument('--do-stacking', action='store_true')
    args = p.parse_args()

    # preprocess
    run([sys.executable, str(src / 'preprocess.py')])

    # tuning (can be long)
    run([sys.executable, str(src / 'optimize_xgb.py'), '--trials', str(args.xgb_trials)])
    run([sys.executable, str(src / 'optimize_cat.py'), '--trials', str(args.cat_trials)])
    run([sys.executable, str(src / 'optimize_te.py'), '--trials', str(args.te_trials)])

    # OOF for stack/ensemble
    run([sys.executable, str(src / 'train_xgb.py')])
    run([sys.executable, str(src / 'train_catboost.py')])
    run([sys.executable, str(src / 'train_te.py')])

    # ensemble search and blend submission
    run([sys.executable, str(src / 'ensemble_te_xgb.py')])
    run([sys.executable, str(src / 'predict_te_xgb_blend.py')])

    if args.do_stacking:
        run([sys.executable, str(src / 'stacking.py')])
        run([sys.executable, str(src / 'predict_stacking.py')])


if __name__ == '__main__':
    main()

