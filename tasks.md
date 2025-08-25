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

