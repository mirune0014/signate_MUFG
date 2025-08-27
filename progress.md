# 進捗報告

閾値最適化やクラス重み調整、ハイパーパラメータ探索、CatBoost・ターゲットエンコーディングの導入により、公開スコアは0.6292となり、順位は101位に向上しました。

## 2025-08-27 更新

- `train.py` に `scale_pos_weight` を導入し、5-fold クロスバリデーションの確率出力を保存。
- `threshold_opt.py` で閾値を0〜1の範囲で探索し、最良の閾値とF1をJSONに記録。
- `predict.py` で最適閾値とクラス重みを用いて LightGBM / XGBoost の推論を実装。

## 2025-08-27 追加

- `preprocess.py` に業種・プログラム別の平均値からの差分や相互作用、分位数ビニングを加え特徴量を拡張。

## 2025-08-28 追加

- `train_resample.py` で SMOTE と RandomUnderSampler を用いた再サンプリング実験を実施し、F1 スコアを `resample_results.json` に保存。
- `train_focal.py` に Focal Loss を実装し、5-fold CV の F1 スコアを `focal_results.json` として記録。

## 2025-08-29 追加

- `optimize.py` と `optimize_xgb.py` で LightGBM / XGBoost のハイパーパラメータを Optuna により探索し、最良パラメータと F1 を `lgb_optuna_results.json` と `xgb_optuna_results.json` に保存。

## 2025-08-30 追加

- `optimize_catboost.py` で CatBoost の `depth`, `learning_rate`, `l2_leaf_reg` などを探索し、最良パラメータと F1 を `cat_optuna_results.json` に記録。
- `train_catboost.py` で最良パラメータを用いた5-fold CV を実施し、平均 F1=0.539 を `cat_cv_results.json` に保存（LightGBM の F1=0.602 と比較し性能は未達）。
## 2025-08-31 追加

- `ensemble.py` を改修し、LightGBM・XGBoost・CatBoost の検証確率を用いた単純平均、F1 重み付け平均、ロジスティック回帰スタッキングを比較し、最良の閾値と F1 を `ensemble_results.json` に記録。
