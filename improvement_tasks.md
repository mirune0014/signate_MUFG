# 改善に向けた考察


- 閾値最適化やクラス重み調整、CatBoost/ターゲットエンコードの実装により、公開スコアは**0.6292**、順位は**101位**まで改善した。
- タスク7で追加特徴量を導入し、タスク8でLightGBMとXGBoostのアンサンブルを評価した。

 - 分類閾値の調整：不均衡データでは確率を0.5で二値化するとF1スコアが伸びにくいため、最適な閾値をグリッドサーチやROC・PRカーブで求めることが推奨されています。
 - クラス重み（scale_pos_weight）の利用：LightGBM公式では二値分類でクラス不均衡を扱う際はis_unbalanceまたはscale_pos_weightを使用するよう推奨しています。

- データリーク対策：前処理やエンコードは必ず訓練データのみで学習し、検証データには学習済みの変換を適用する必要があります。パイプラインや交差検証内での処理が重要です。
- CatBoostの活用：CatBoostはカテゴリ変数を自然に扱うよう設計されており、順序付きブースティングやオブリビアスツリーなどの工夫で最小限のチューニングで高性能を発揮します。
- 特徴量設計とターゲットエンコーディング：既存の特徴量に加え、業種・州・企業形態別のデフォルト率などをターゲットエンコーディングで追加すると性能向上が期待できます。リークを防ぐため、各fold内の訓練データでエンコーディングし検証データへ適用します。
- ハイパーパラメータ最適化の拡充：Optunaなどで探索するパラメータを増やし、num_boost_round、max_depth、min_child_samples、lambda_l1、lambda_l2、scale_pos_weightなどを広範囲に試す。
- 他モデルの検討：XGBoostやCatBoostのハイパーパラメータチューニングを行い、最終的にアンサンブルで組み合わせる。

## タスクリスト

| # | タスク名 | 内容 |
|---|---|---|
| 1 | ベースライン評価 | 現行のLightGBM・XGBoostモデルで交差検証を実行し、F1スコアと予測確率を確認する。 |
| 2 | 閾値最適化 | 交差検証予測確率からF1が最大となる分類閾値を探索する。 |
| 3 | クラス重み調整 | LightGBMにis_unbalanceまたはscale_pos_weightを設定し、各種値でF1を評価する。 |
| 4 | ハイパーパラメータ探索 | OptunaでLightGBMのパラメータ範囲を拡張し、より広い探索を行う。XGBoostについても同様にチューニングする。 |
| 5 | CatBoost実験 | CatBoostClassifierでカテゴリ変数をそのまま扱い、パラメータ（iterations, depth, learning_rate等）をチューニングしながら交差検証する。 |
| 6 | ターゲットエンコーディング | 業種・州などの高基数カテゴリに対し、クロスバリデーション内でターゲット平均を計算して特徴量として追加し評価する。 |
| 7 | 追加特徴量 | 金利差の区分や債務比率、保証率などの新規特徴量やカテゴリ組み合わせを設計し評価する。 |
| 8 | アンサンブル評価 | 改善された各モデルの予測確率を平均や重み付きで組み合わせ、最終的なF1スコアを計算する。 |

## 手順書（各タスクの実施方法）

### 共通準備

1. **環境構築**：`python -m venv venv`で仮想環境を作成し、`pip install -r requirements.txt`で必要なライブラリ（pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, optuna, imbalanced-learn など）をインストールします。
2. **データ確認**：`data/input/train.csv`と`test.csv`をロードし、LoanStatusのクラス分布を確認して不均衡比率を把握します（既存の`src/eda.py`でも確認可）。
3. **前処理のパイプライン化**：データリークを防ぐため、ColumnTransformerやターゲットエンコーディングを含めた処理を`sklearn.pipeline.Pipeline`内に実装し、交差検証の中でfit / transformを行います。

### タスク1：ベースライン評価

- `python src/preprocess.py`で既存の前処理を実行し、`train_preprocessed.npz`を生成します。
- `python src/train.py`と`python src/train_xgb.py`を実行して、LightGBM・XGBoostの5-fold F1スコアと各foldの予測確率を取得します。
- 結果を記録し、今後の実験との比較基準とします。

### タスク2：閾値最適化

- タスク1で得た各foldの検証データに対する予測確率と実際のラベルを保存します。
- 0〜1の範囲でステップ0.01程度の閾値を設定し、各閾値で二値化してF1スコアを計算します。
- fold毎に最適な閾値を求め、その平均または中央値を最終モデルの閾値として採用します。
- 評価コードでは、`predict_proba()`で得られる確率を使い、閾値を変更できるように関数化します。

### タスク3：クラス重み調整

- クラスの不均衡比率をもとに`scale_pos_weight = (negative_samples / positive_samples)`を計算します。
- LightGBMのパラメータに`is_unbalance=True`または`scale_pos_weight`を追加し、交差検証でF1を評価します。
- `scale_pos_weight`を周辺±20%ほど増減させて数値を変え、最適な値を探索します。

### タスク4：ハイパーパラメータ探索

- **LightGBM**：`src/optimize.py`を拡張し、`max_depth`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l1`, `lambda_l2`, `is_unbalance`または`scale_pos_weight`等を試すよう`trial.suggest_*`を追加します。`n_trials`も増やして（例: 50〜100回）最適なパラメータを探索します。
- **XGBoost**：新規に`optimize_xgb.py`などを作成し、`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`をOptunaで探索します。
- 最適なパラメータとスコアをJSONファイルに記録し、後続の学習・推論で利用できるようにします。

### タスク5：CatBoost実験

- 前処理では数値変換やログ変換等のみを行い、カテゴリ列はそのまま保持します。カテゴリ列名を`cat_features`としてCatBoostに渡します。
- `from catboost import CatBoostClassifier`をインポートし、`iterations`, `depth`, `learning_rate`, `l2_leaf_reg`等をOptunaまたはGridSearchCVでチューニングします。
- CatBoostは内部でカテゴリ変数をエンコードするため、One-Hotやターゲットエンコーディングを別途行う必要はありません。
- 5-fold層化交差検証でF1を評価し、最も良いパラメータとスコアを保存します。

### タスク6：ターゲットエンコーディング

- `category_encoders`ライブラリ（例：`pip install category_encoders`）を利用します。
- `LoanProgram`, `NaicsSector`, `BusinessType`, `BorrState`等のカテゴリについて、各foldの訓練データで目標平均（デフォルト率）を計算し検証データに適用するターゲットエンコーディングを実装します。
- このエンコード処理も`sklearn.pipeline`に組み込み、`cross_val_score`でF1を評価します。
- LightGBMやXGBoost、CatBoostに入力し、それぞれのスコアを記録して他の手法と比較します。

### タスク7：追加特徴量

- 現行の`preprocess.py`で作成しているログ変換や比率に加え、以下の特徴量を検討します（例）：
  - 州ごとの融資件数やデフォルト率（ターゲットエンコードの延長として集約特徴量）。
  - 業種別平均金利や平均融資額との差。
  - 期間比率（`TermInMonths`と`GrossApproval`, `SBAGuaranteedApproval`の組合せ）。
  - カテゴリ同士の組み合わせ（`BorrState×BusinessType`など）の交互作用特徴。
- 新規特徴量を追加した前処理を実装し、タスク4〜6のモデルで評価します。

### タスク8：アンサンブル評価

- 各モデル（LightGBM, XGBoost, CatBoost, ターゲットエンコード版など）の検証用予測確率を保存します。
- 単純平均や重み付き平均（Optunaで重みを最適化する等）によりアンサンブルし、F1スコアを計算します。
- ベースラインや単一モデルより改善されているか確認し、最終提出モデルに採用する構成を決定します。

