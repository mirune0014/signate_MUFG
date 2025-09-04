# Competition Postmortem (MUFG, finished 2025-09-02)

- Final private score: 0.65397923 (F1), rank: 17/—
- Public peak (representative): 0.64007092 with 3-model blend (TE/XGB/Cat), global thr≈0.40
- Target was 0.66; time ran out to close the last ~0.006–0.01 gap.

## What Moved The Needle

- Feature engineering grounded in lending risk:
  - Interest burden & term: Annual/MonthlyInterestBurden, InterestRatePerTerm, RateTerm (binned rate × term). Helped separability and improved XGB/Cat OOF.
  - Coverage & collateral: ApprovalRatio, GuaranteeGapRatio, HasCollateral, LongTermNoCollateral, DebtServiceToGuarantee. Made defaults more detectable at long-tenor/high-rate.
  - Macro/temporal controls: YearInterestDiff, YearApprovalDiff, YearProgram*Diff, YearCentered, Post2010/2015. Captured regime shifts; stabilized models.
  - Scale proxies: Jobs_log, GrossPerJob, GuaranteePerJob. Added nonlinearity that trees exploited well.
- Model/ensemble:
  - CatBoost tuned (iterations≈640, depth=4, lr≈0.039) + XGB tuned + TE-LGB → complementary biases.
  - 3-model weight search + threshold sweep → best offline F1≈0.6385, best LB≈0.6401.
- Threshold care:
  - Global thr around 0.40 consistently outperformed 0.5. Micro-tuning（±0.01〜0.02）worth trying on LB.

## What Did Not Transfer (OOF→LB)

- Group thresholds (best OOF on NaicsSector: +0.0115) dropped on LB (0.6386 < 0.6401).
  - Likely reasons: sector mix shift between public/private; small groups; threshold overfitting.
- Stacking (LR meta) and high-confidence pseudo-labels: both underperformed on LB.
  - Probable causes: metric thresholding sensitivity; calibration mismatch; pseudo labels amplifying bias.

## Validation & Calibration Lessons

- Use time-aware CV from day 1 when ApprovalFiscalYear exists. Our later time-aware threshold check confirmed global thr stability (0.40), but earlier adoption would reduce LB drift risks.
- Prefer fold-wise TargetEncoding only; avoid any leakage via global fitting. We kept TE per fold but should log-check consistently.
- Persist test-time probabilities and sweep a narrow threshold band around OOF optimum (done). For LB, 0.39/0.41 variants were useful.
- When trying group thresholds, enforce:
  - Minimum support with shrinkage toward global thr.
  - Monotone smoothing or hierarchical pooling to avoid spiky per-group thresholds.

## Feature Lessons (Keep/Extend)

- Keep: interest burden, guarantee/collateral, rate×term, regime diffs, jobs-based proxies.
- Extend: carefully selected cross features — (Collateral×TermBucket), (BusinessType×Collateral), (Sector×BusinessAge), (YearProgram×VariableRate) — evaluate in small batches.
- Consider monotone constraints (e.g., w.r.t. InterestRate, Term) in XGB/LightGBM where it matches domain logic; may help generalization.

## Modeling/Ensembling Lessons

- CatBoost often complements XGB on mixed tabular with many categoricals; tuned shallow trees (depth 4–6) worked best here.
- Light seed bagging at inference (3 seeds) added mild stability without big runtime cost.
- Stacking helps if base models are diverse and calibrated; otherwise thresholded F1 may penalize slight miscalibration.

## Experiment Management

- Kept an experiments log (docs/EXPERIMENTS.md) and added SUMMARY/NEXT_STEPS.
- Batch runner (src/run_experiments.py) accelerated reproducibility.
- Saved blend probabilities (for threshold variants) and OOF arrays (for ensemble search).

## If We Had More Time (Actionable)

1) Expand Optuna trials (XGB 250–300, Cat 200+, TE 120+) with early-stopped CV.
2) Switch CV to forward-time split across all models and re-tune thresholds on most recent folds.
3) Add 3–5 more curated interactions; keep feature budget tight to avoid noise.
4) Try probability calibration (isotonic/Platt) before F1 thresholding; re-check LB alignment.
5) Heavier seed bagging or light snapshot ensembling for XGB/Cat.
6) Explore monotone constraints on key monotonic features.

## Reusable Checklist (Next Competition)

- Problem framing: map features to real drivers; write 3–5 hypotheses first.
- CV protocol: decide stratification and time-awareness; test at least 2 schemes early.
- Leakage control: fold-wise encoders, no target in global stats; audit.
- Thresholding: always sweep around OOF optimum; consider group thresholds with shrinkage.
- Ensembles: start with 2–3 diverse tabular models; weight search on OOF; avoid heavy stacking unless calibrated.
- Logging: one experiments log + labeled submissions; save probs for later threshold probes.

