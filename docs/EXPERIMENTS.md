# Experiments Log

This document tracks experiments, changes, CV scores, and LB results. Keep entries short and consistent for quick review and reproducibility.

## How To Log
- Create a new entry under 窶廢ntries窶・with:
  - ID: short unique id like `EXP-YYYYMMDD-N`
  - Changes: bullet list of code/config changes
  - Data/CV: CV scheme and key params
  - Result (CV): mean F1 and notes
  - Result (LB): fill after submission
  - Artifacts: key files (paths to JSON/CSV produced)

## Entries

### EXP-BASELINE-01
- Changes:
  - Preprocess with engineered features (`src/preprocess.py`)
  - LGBM 5-fold CV with `scale_pos_weight` and OOF save (`src/train.py`)
  - Threshold optimization over [0.00, 1.00] step 0.01 (`src/threshold_opt.py`)
  - Equal-weight LGBM+XGB inference (`src/predict.py`)
- Data/CV:
  - CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
  - LGBM: 100 iters, default params + pos_weight
- Result (CV): mean F1=0.6031; best F1@thr=0.6160 (thr=0.62)
- Result (LB): TBD (fill after submit)
- Artifacts:
  - `data/output/oof_preds.npy`
  - `data/output/cv_results.json`
  - `data/output/threshold_results.json`
  - `data/output/submit.csv`

### EXP-ENSEMBLE-OPT-01
- Changes:
  - Path refactor to be relative (portable)
  - Add imbalance handling to XGB and CV loops
  - Grid-search ensemble weight + threshold on OOF (`src/ensemble.py`)
  - Inference reads best weight/threshold if present (`src/predict.py`)
- Data/CV:
  - Same 5-fold CV; grid: weight 0.0窶・.0 step 0.05, thr 0.00窶・.00 step 0.01
- Result (CV): best F1=0.6162 (weight=0.60, thr=0.48)
- Result (LB): 0.63006 (file: data/output/submit.csv)
- Artifacts:
  - `data/output/ensemble_results.json`
  - `data/output/ensemble_oof_preds.npz`

### EXP-TE-CATBOOST-01
- Changes:
  - Target encoding inside CV (`src/train_te.py`)
  - CatBoost baseline with categorical features (`src/train_catboost.py`)
- Data/CV:
  - Same 5-fold CV
- Result (CV):
  - TE LGBM mean F1=0.6221 (`data/output/te_cv_results.json`)
  - CatBoost mean F1=0.5802 (`data/output/catboost_cv_results.json`)
- Result (LB): 0.63149 (file: data/output/submit_te.csv)

### EXP-STACK-01
- Changes:
  - Saved OOF for XGB/TE/Cat to enable stacking
  - Added `src/stacking.py` (meta LR on OOF) and `src/predict_stacking.py`
- Data/CV:
  - 5-fold CV; meta LogisticRegression(class_weight=balanced)
- Result (CV): best F1=0.6283 (`data/output/stacking_results.json`)
- Result (LB): TBD (file: data/output/submit_stack.csv)

### EXP-BLEND-01
- Changes:
  - Multi-model weighted blend (LGBM/XGB/TE/Cat) with weights+threshold search on OOF
  - Inference via `src/predict_blend.py`
- Data/CV:
  - 5-fold CV; select top-3 models by single-model F1 for blending
- Result (CV): see `data/output/ensemble_results.json`
- Result (LB): 0.63149 (file: data/output/submit_blend.csv)
  - Note: No improvement over TE best; TE remains SOTA in this repo

### EXP-TE-OPT-02
- Changes:
  - Added `src/optimize_te.py` to tune TE smoothing/min_samples and LGBM params
  - Updated `src/predict_te.py` to consume `optuna_te_best.json`
- Data/CV:
  - 5-fold CV with OOF threshold sweep inside objective
- Result (CV): best F1=0.6385 (`data/output/optuna_te_best.json`)
- Result (LB): 0.64163 (Rank 72) (file: data/output/submit_te.csv)

## Next Candidates
- Extended Optuna spaces for LGBM/XGB
- Additional interaction features and robust pipelines
- Stacking meta-model with OOF predictions

### EXP-FE-V2-Preprocess
- Changes:
  - Added rich domain features in `src/preprocess.py`:
    - Group deltas: Sector/Program/BusinessType × (Interest, Gross)
    - Frequency encodings: NaicsSector, BusinessType, Subprogram, Fixed/Variable, Revolver, Collateral
    - Per-job metrics: GrossPerJob, GuaranteePerJob; MonthlyPayment_log; flags (HighInterest, LongTerm)
  - Trained LGBM with early stopping (AUC eval) in `src/train.py`
- Result (CV): mean F1 ≈ 0.6072; best F1@thr ≈ 0.6078 (`cv_results.json`, `threshold_results.json`)
- Result (LB): TBD (file: data/output/submit.csv)

### EXP-FE-V2-TE
- Changes:
  - Mirrored the domain features in TE pipeline (`src/train_te.py`, `src/predict_te.py`)
  - TE remains CV-safe; threshold priority uses TE-optimized > ensemble > baseline
- Result (CV): see `data/output/te_cv_results.json`
- Result (LB): TBD (file: data/output/submit_te.csv)

### EXP-TE-XGB-BLEND-02
- Changes:
  - Added interaction categories for TE: Sector×Program, Sector×BusinessType
  - Two-model blend optimizer `src/ensemble_te_xgb.py` (TE vs XGB) and inference `src/predict_te_xgb_blend.py`
- Data/CV:
  - OOF-based grid search: weight_te∈[0,1] step 0.01; threshold∈[0,1] step 0.01
- Result (CV): best F1=0.6313, weight_te=0.25, thr=0.61 (`data/output/ensemble_te_xgb.json`)
- Result (LB): TBD (file: data/output/submit_te_xgb.csv)
## 2025-09-01: Feature-enriched TE/XGB/Cat + Blending (this session)

- Objective: Raise LB toward 0.66 by adding reality-grounded features and improving ensembles.
- Hypothesis: Risk aligns with (a) interest burden vs term, (b) guarantee/coverage and collateral, (c) sector/program × year macro conditions, (d) business size proxies. Encoding these interactions improves separability and calibration.
- Changes:
  - Added engineered features to `preprocess.py`, `train_te.py`, `predict_te.py`, `predict_te_xgb_blend.py` (interest burden, guarantee gap, term buckets, rate×term, year/program diffs, frequency encodings, temporal flags, collateral/variable flags, jobs log, etc.).
  - Implemented `optimize_xgb.py`/`optimize_te.py` CLI for trials; added `optimize_cat.py` for CatBoost tuning with OOF export.
  - Upgraded `train_catboost.py` to use tuned params and richer features; `ensemble_te_xgb.py` can blend 2 or 3 models; `predict_te_xgb_blend.py` supports 3-way blending.
  - Added `run_experiments.py` batch runner.
- Offline OOF (F1):
  - XGB (Optuna, 120 trials): ~0.6326
  - TE (Optuna applied at inference): ~0.546 (TE alone weaker but helpful for blend)
  - CatBoost (Optuna 20 trials, best): ~0.6372 （iterations≈640, depth=4, lr≈0.0387）
  - Blend (TE+XGB+Cat, weight/threshold grid): ~0.6385（weights≈[TE 0.10, XGB 0.25, Cat 0.65], thr≈0.40）
- LB result:
  - submit_te_xgb.csv = 0.63195691（ユーザー報告）
  - Next submissions: submit_te_xgb.csv（再作成, tuned XGB/Cat/TE 反映後）, submit_stack.csv（メタ学習）→LB待ち（提出後ここに数値追記）

## Next Plan

1) Increase Optuna trials (XGB 200–300, Cat 200+, TE 100+) to tighten generalization.
2) Try time-aware CV (train: past, valid: future by `ApprovalFiscalYear`) for threshold robustness.
3) Add interaction features: (Collateral × TermBucket), (BusinessType × Collateral), (Sector × BusinessAge), (YearProgram × VariableRate).
4) Pseudo-labeling: high-confidence test preds (p>0.95 or <0.05) to augment train for XGB/Cat; compare OOF & LB.
5) Stacking meta with LogisticR (current `stacking.py`), then try calibrated threshold per-year (ablation vs global threshold).

## 2025-09-02: CatBoost最適化 + 3モデルブレンド + 疑似ラベルXGB

- Objective: さらにLB改善。CatBoostの最適化と、3モデルブレンド比率最適化、擬似ラベルXGBによる境界強化。
- Hypothesis: CatBoostはカテゴリ×連続の相互作用を素で捉えやすく、ブレンド寄与が大きい。高確信サンプルの擬似ラベルで決定境界をシャープにできる。
- Changes:
  - `optimize_cat.py`（20 trials）でCatBoost最適化・OOF保存。
  - `train_catboost.py`を最適化パラメータ読込に対応、特徴強化。
  - `ensemble_te_xgb.py`で3モデル（TE/XGB/Cat）重み探索。
  - `pseudo_label_xgb.py`で高確信(p≥0.98 or ≤0.02)の擬似ラベル学習と提出作成。
- Offline OOF (F1):
  - CatBoost best: ~0.6372
  - 3-model blend: ~0.6385（weights≈[0.10, 0.25, 0.65], thr≈0.40）
- Submissions to check on LB:
  - `submit_te_xgb.csv`（3モデルブレンド, thr≈0.40）
  - `submit_stack.csv`（ロジスティック・スタッキング）
  - `submit_xgb_pseudo.csv`（XGB擬似ラベル）
- LB: 提出後に追記してください（例: 0.6319→0.64x/0.65x …）。

## 2025-09-02: サブグループ別しきい値最適化（OOF）

- Objective: OOFブレンド確率に対して、グループごとに最適しきい値を設定しF1をさらに押し上げる。
- Hypothesis: 産業や年度、事業形態によって基準が異なるため、単一のしきい値よりサブグループ別のほうがF1が改善する。
- Changes:
  - `threshold_group_opt.py`: OOFブレンド（TE/XGB/Catの重みを`ensemble_te_xgb.json`から再現）に対し、各候補グループ（Subprogram/NaicsSector/ApprovalFiscalYear/BusinessType）で最適しきい値を推定（最小支持=150）。最良グループを採用し`threshold_group_results.json`に保存。
  - `predict_te_xgb_blend.py`: ブレンド確率`test_blend_prob.npy`を保存。
  - `predict_group_threshold.py`: テストにグループ別しきい値を適用し`submit_te_xgb_group.csv`を生成。
- OOF結果:
  - Global thr（0.40）: F1 ≈ 0.6385
  - Best group = `NaicsSector`（min_support=150）: F1 ≈ 0.6500（+0.0115）
  - `threshold_group_results.json`に採択グループと各しきい値を保存済み。
- 提出物:
  - `submit_te_xgb_group.csv`（NaicsSector別しきい値）→ LB反映後にここへ数値追記。



submit_stack.csv 0.6260504201680672
submit_xgb_pseudo.csv 0.6235521235521236
submit_te_xgb.csv	0.6400709219858155	


ubmit_te_xgb_group.csv	0.6385650224215248