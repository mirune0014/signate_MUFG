## 提出までのタスク表

| # | 進捗 | タスク | 詳細手順 |
|---|------|--------|----------|
| 1 | ✅ | **開発環境の構築** | 1) Python 3.9 以降の仮想環境を作成<br>2) `pip install` で `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `optuna` など主要ライブラリを導入 |
| 2 | ✅ | **データ取得・配置** | `data/input` にある `train.csv`・`test.csv`・`sample_submit.csv` をプロジェクトのデータディレクトリへ配置（すでに配置済みであれば確認のみ） |
| 3 | ✅ | **探索的データ分析 (EDA)** | 1) `pandas` で CSV を読み込み、行数・列数・型を確認<br>2) `describe()` やヒストグラムで数値分布、`value_counts()` でカテゴリ分布を把握<br>3) 目的変数 `LoanStatus` の割合を算出しクラス不均衡を確認 |

| 4 | ✅ | **前処理** | 1) 欠損チェック（今回のデータは欠損なしだが再確認）<br>2) カテゴリ変数を列挙（例: `LoanProgram`, `NaicsSector`, `BusinessType`, `BorrState`, `IsFranchise` など）<br>3) 数値変数のスケーリング（必要に応じて標準化/対数変換） |
| 5 | ✅ | **特徴量エンジニアリング** | 1) カテゴリ変数に対し One-Hot または Target Encoding を適用<br>2) 初期金利等で金利差分・金利区分を作成<br>3) 融資額の対数変換、期間あたりの支払額など活用量を作成<br>4) 事業年数カテゴリから数値化（例: “Startup”→0年、“3-5 years”→4年） |
| 6 | ✅ | **学習データと評価設定** | 1) 特徴量行列 X と目的変数 y (`LoanStatus`) を分割<br>2) `StratifiedKFold` で層化 k-fold（推奨 k=5）を設定 |
| 7 | ✅ | **ベースラインモデル構築** | 1) まず `LightGBM` でモデル作成<br>2) F1 スコアを指標に交差検証<br>3) しきい値調整（0.5 以外でも F1 最大化を探索） |
| 8 | ✅ | **ハイパーパラメータ最適化** | 1) `optuna` 等で `num_leaves`, `max_depth`, `learning_rate`, `class_weight` などを探索<br>2) 各 fold の F1 スコア履歴を保存 |
| 9 | ✅ | **他モデルの検討（任意）** | `XGBoost` を学習し、交差検証で LightGBM とスコア比較 |
|10 | ✅ | **モデルアンサンブル（任意）** | LightGBM と XGBoost の予測確率を平均化して F1 改善を検討 |
|11 | ✅ | **最終モデル学習** | 最も性能の高かった設定で全学習データを再学習 |
|12 | ✅ | **推論・提出ファイル作成** | 1) `test.csv` を前処理・特徴量変換後、最終モデルで予測<br>2) `sample_submit.csv` の形式（`Id`, `LoanStatus`）に合わせて出力<br>3) 生成した CSV を `submit.csv` として保存 |
|13 | ✅ | **再現性とドキュメント** | 1) スクリプトをリポジトリに整理<br>2) `README` に環境情報・実行手順を記載<br>3) 推論手順を `explain.md` に追記し提出 |


## モデル改善のためのタスク表

| # | 進捗 | タスク | 詳細手順 |
|---|------|--------|----------|
| 14 | ☐ | 閾値最適化とクラス不均衡対応 | 1) 学習データから正例/負例比を計算し `scale_pos_weight` を決定（例: 件数比率）<br>2) `train_weighted.py` を参考に `LightGBM`/`XGBoost` の訓練時に `scale_pos_weight` を引数として渡せるよう `train.py` と `predict.py` を改修<br>3) 5-fold クロスバリデーションの出力確率を保存し、0〜1 の範囲で 0.01 刻みなど細かく閾値を変えて F1 スコアを評価するスクリプトを作成し（`threshold_opt.py` 等）、最良の閾値と F1 を JSON に保存<br>4) `predict.py` でこの最適閾値を使用し二値予測を生成 |
| 15 | ☐ | 特徴量エンジニアリングの強化 | 1) 業種（`NaicsSector`）、州（`BorrState`）や融資プログラムごとに平均デフォルト率や平均融資額・平均期間を集計し、各サンプルとの差分や比率を新規特徴量として追加<br>2) 高基数カテゴリ同士の組み合わせ（例: `BorrState × BusinessType` や `NaicsSector × LoanProgram`）を文字列結合して相互作用特徴量を作成<br>3) 融資額・期間・返済額などの数値列に対し z-score 正規化や分位数ビニングを試し、外れ値の影響を抑える<br>4) これら処理を `preprocess.py` もしくは新規スクリプトで実装し、クロスバリデーション内では各 fold の学習データのみで集計/正規化を学習して検証データに適用しデータリークを防止 |
| 16 | ☐ | 再サンプリングと損失関数の検証 | 1) 不均衡データのバランスを改善するため、`imblearn` ライブラリの `SMOTE`（オーバーサンプリング）や `RandomUnderSampler` を `Pipeline` に組み込み、`LightGBM`/`XGBoost` に入力する実験を行う<br>2) `LightGBM`/`XGBoost` で `Focal Loss` など不均衡対応の損失関数を選択できるようカスタム損失を実装し、標準損失関数との比較を行う<br>3) 各手法について 5-fold CV で F1 を評価し、効果の高いものを記録 |
| 17 | ☐ | ハイパーパラメータ探索の拡張 | 1) `optimize.py` の探索範囲を拡張し、`LightGBM` の `num_leaves`, `max_depth`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2` などを対象にする<br>2) `XGBoost` についても `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `learning_rate`, `n_estimators` などを `optuna` で探索する新規スクリプトを作成<br>3) 各モデルの最良パラメータと F1 を JSON で保存し、再現性のため乱数シードを固定 |
| 18 | ☐ | CatBoost の導入と調整 | 1) `train_catboost.py` を拡張し、カテゴリ列をそのまま入力として `CatBoostClassifier` を学習させる<br>2) `CatBoost` の `depth`, `learning_rate`, `l2_leaf_reg`, `iterations` などを GridSearch または `Optuna` でチューニング<br>3) 5-fold CV で F1 を測定し、`LightGBM`/`XGBoost` と比較<br>4) `CatBoost` のベストモデルを保存し、アンサンブル候補に加える |
| 19 | ☐ | アンサンブル戦略の最適化 | 1) `ensemble.py` を改修し、複数モデルの予測確率を単純平均するだけでなく、各モデルの検証 F1 に基づく重み付け平均を行えるようにする<br>2) `LightGBM`・`XGBoost`・`CatBoost` の出力確率を特徴量として用い、メタモデル（ロジスティック回帰や線形 SVM など）で二次学習を行うスタッキング手法を実装<br>3) 各アンサンブル手法について交差検証で F1 を比較し、最良の手法と閾値を記録 |
| 20 | ☐ | パイプライン化とデータリーク防止 | 1) `sklearn` の `Pipeline` と `ColumnTransformer` を用いて、欠損処理・スケーリング・ターゲットエンコーディング・集約特徴量作成からモデル学習までを一つのパイプラインにまとめる<br>2) クロスバリデーション内で各 fold の学習データのみから前処理を学習し、検証データには transform のみを適用してリークを防ぐ<br>3) ターゲットエンコーディングでは `category_encoders.TargetEncoder` の `cv` 引数や `Leave-One-Out` を活用し、 `smoothing` パラメータで過学習を抑制 |
| 21 | ☐ | 検証分割と再現性の見直し | 1) 公開 leaderboard とテストデータの分布差を考慮し、時間や地域を考慮した `GroupKFold` や `TimeSeriesSplit` を検討<br>2) すべてのスクリプトで `random_state` を固定し、分割やモデルの乱数要素を制御することで実験の再現性を高める<br>3) 検証結果や分割情報をログとして `data/output` に保存 |
| 22 | ☐ | 最終モデルの構築と提出更新 | 1) 上記タスクで得られた最良の前処理・特徴量・ハイパーパラメータ・クラス重み・閾値を用いて、`LightGBM`・`XGBoost`・`CatBoost` を全学習データで再学習<br>2) 最良のアンサンブル手法を適用してテストデータの予測確率を生成し、閾値で二値化<br>3) `sample_submit.csv` に従った提出用ファイル `submit.csv` を出力し、`predict.py` も更新して誰でも同じ手順で提出ファイルを作れるようにする |
