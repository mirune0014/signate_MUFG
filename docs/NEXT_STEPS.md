# Next Steps (As of 2025-09-03)

- Goal: push LB from 0.6401 → 0.645–0.650+ with robust generalization.

## Immediate (quick wins)
- Submit threshold variants around 0.40 to probe LB optimum:
  - Files ready: `submit_te_xgb_thr39.csv`, `submit_te_xgb_thr41.csv` (optionally `thr38/42`).
- Keep `submit_te_xgb.csv` (3-model blend) as baseline for daily reference.

## Short Term (hours)
- Increase Optuna trials:
  - XGB: `python src/optimize_xgb.py --trials 200`
  - Cat: `python src/optimize_cat.py --trials 200`
  - TE : `python src/optimize_te.py  --trials 120`
- Rebuild OOF + blend + submit:
  - `python src/train_xgb.py && python src/train_catboost.py && python src/train_te.py`
  - `python src/ensemble_te_xgb.py && python src/predict_te_xgb_blend.py`

## Medium Term (day)
- Time-aware CV (ApprovalFiscalYear forward split) to improve LB alignment.
- Feature ablations (small batches):
  - (Collateral × TermBucket), (BusinessType × Collateral),
  - (Sector × BusinessAge), (YearProgram × VariableRate).
- Seed bagging breadth for Cat/XGB (3→5 seeds) for stabler submissions.

## Utilities
- Group thresholds (NaicsSector best in OOF):
  - Optimize: `python src/threshold_group_opt.py --min-support 150`
  - Predict: `python src/predict_group_threshold.py` → `submit_te_xgb_group.csv`
- Time-segment threshold:
  - `python src/threshold_time_opt.py` (threshold stored for blend inference)
- Batch runner:
  - `python src/run_experiments.py --xgb-trials 200 --cat-trials 200 --te-trials 120 --do-stacking`

