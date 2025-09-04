# Competition Tips (Reusable)

## Setup & Workflow
- **Batch runner:** use a `run_experiments.py` scaffold to automate preprocess → tuning → OOF → blend → submit.
- **Artifacts:** persist OOF arrays and test probabilities; they unlock ensembles and threshold A/B without re-training.
- **Logging:** maintain a single `EXPERIMENTS.md` with date, purpose, hypothesis, change, OOF/LB.

## Feature Engineering
- **Start from reality:** tie features to causal/risk hypotheses; prefer low-leakage aggregates (e.g., within-train/global-no-target means).
- **Interactions:** add small, high-signal crosses; avoid combinatorial explosions; ablate in small batches.
- **Stability:** center to regimes (year/program), use ratios (coverage), and logs for heavy-tailed counts.

## Validation & Thresholding
- **Time-aware CV:** when temporal notion exists, forward split early; compare with stratified.
- **Target encoding:** fold-wise only; consider smoothing/min_samples.
- **Threshold sweep:** always probe ±0.02 around OOF-opt threshold on LB. Save probs to generate CSV variants.
- **Group thresholds:** only with minimum support + shrinkage to global; watch for LB drift.
- **Calibration:** if metric uses thresholded probabilities, consider isotonic/Platt on OOF.

## Modeling
- **Diversity:** combine XGB/LightGBM/CatBoost; keep depths modest; regularize.
- **Tuning:** use Optuna with early stopping; cap search spaces; log best thresholds.
- **Bagging:** 3–5 seeds for inference stability; consider snapshot/SGDR for NN (if any).
- **Stacking:** useful when base models are calibrated and complementary; otherwise simple weighted blends may win.

## LB Alignment
- **Holdout recency:** choose thresholds on the most recent fold(s); verify distribution shifts (counts per group/year).
- **Monitor gap:** track OOF→LB deltas for each submission to detect overfitting quickly.

## Ops & Hygiene
- **Reproducibility:** fix seeds; record library versions; commit code and configs.
- **Budget:** prefer few, high-quality features over many noisy ones; watch train time.
- **Stop rules:** if LB stops improving with a branch, revert to last best and pivot.

