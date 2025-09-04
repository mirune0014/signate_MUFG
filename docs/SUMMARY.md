# Project Summary and Score Timeline

This document summarizes the experiments, major feature additions, and their observed impacts (OOF/LB).

## Data/Metric
- Binary target: `LoanStatus` (1=Default)
- Primary metric: F1 (public LB)

## Baseline and Milestones
- Baseline (user context): `submit_te_xgb.csv` ≈ 0.6385 → initial LB 0.6319
- Current best LB: `submit_te_xgb.csv` (3-model blend, thr≈0.40) = 0.6400709219858155

## Model/Features Overview
- Models: XGB (OHE preproc), LGBM with TargetEncoding (TE), CatBoost (raw + engineered), simple stacking (LR)
- Ensembles: weight search over OOF (TE/XGB/Cat), global threshold sweep; optional group/time thresholds

## Key Feature Themes (engineered)
- Interest burden: `AnnualInterestBurden`, `MonthlyInterestBurden`, `InterestRatePerTerm`, `HighInterestFlag`
- Coverage & collateral: `ApprovalRatio`, `GuaranteeGapRatio`, `HasCollateral`, `LongTermNoCollateral`, `DebtServiceToGuarantee`
- Term × Rate: `RateTerm` (binned rate × term bucket), `TermInYears`
- Macro/temporal: `YearInterestDiff`, `YearApprovalDiff`, `YearProgram*Diff`, `YearCentered`, `Post2010/2015`
- Sector/program interactions & frequencies: `ProgramBusinessType`, `Sector*`, frequency encodings including `YearProgram`
- Size proxies: `Jobs_log`, `GrossPerJob`, `GuaranteePerJob`

## Score Table (known points)

| Change / Submission | Notes | OOF F1 | LB F1 |
|---|---|---:|---:|
| XGB (optuna, tuned) | preprocessed OHE | ~0.6326 | – |
| TE (tuned at inference) | LGBM+TargetEncoder | ~0.546 | – |
| CatBoost (optuna 20t) | engineered features | ~0.6372 | – |
| 3-model blend (TE/XGB/Cat) | weight/threshold grid, thr≈0.40 | ~0.6385 | 0.6401 |
| Group thresholds (NaicsSector) | best OOF +0.0115 over global | ~0.6500 | 0.6386 |
| Stacking (LR meta) | lgb/xgb + te/cat | – | 0.6261 |
| XGB pseudo-label | p≥0.98/≤0.02 augment | – | 0.6236 |

Notes:
- OOF improvements with group thresholds did not transfer to LB in first attempt; retained as optional path.
- Threshold micro-tuning around 0.40 is prepared; submit `thr39` and `thr41` variants to probe LB optimum.

## Files of Interest
- Features & preprocessing: `src/preprocess.py`, `src/train_te.py`, `src/predict_te.py`, `src/predict_te_xgb_blend.py`
- Tuning: `src/optimize_xgb.py`, `src/optimize_cat.py`, `src/optimize_te.py`
- OOF/Ensemble: `src/train_xgb.py`, `src/train_catboost.py`, `src/train_te.py`, `src/ensemble_te_xgb.py`
- Threshold tooling: `src/threshold_group_opt.py`, `src/predict_group_threshold.py`, `src/threshold_time_opt.py`, `src/make_threshold_variants.py`
- Batch pipeline: `src/run_experiments.py`
- Logs: `docs/EXPERIMENTS.md`

