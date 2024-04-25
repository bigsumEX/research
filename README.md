# NOTEARSとSEMによる構造学習
<div id="top"></div>

## 使用技術一覧

<!-- シールド一覧 -->
<p style="display: inline">
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
</p>

## 分析の流れ
1. データ取得
2. データ前処理
3. Optunaによるハイパーパラメータのベイズ最適化
4. NOTEARSによる構造学習
 
## 分析方法の詳細
### データ前処理
- 訓練データとテストデータに分割
    - 訓練:7割, テスト:3割
- データの正規化
    - モチベーション: データの大小が極端なので足並みを揃える
    - 手法: MinMaxScalerによる0~1範囲での正規化

### Optunaによるハイパーパラメータのベイズ最適化
- pytorch.from_padasの学習に用いるパラメータを最適化する

- 最適化の評価指標: RMSEA
    - RMSEA: 共分散構造分析(SEM)によるモデルとデータの適合度を表現する評価指標
- 探索するパラメータ: lasso_beta, ridge_beta, threshold

### NOTEARSによる構造学習
- RMSEAを最小化するように構造を学習する
- 閾値は任意(目的に沿って設定)

## 使い方


## ディレクトリ構成
### 各ディレクトリについて
- data: 実行に必要なデータを格納するディレクトリ
- functions: 関数として定義した各機能が格納されているディレクトリ
- work: 検証用ディレクトリ
- output: 出力した結果が格納されているディレクトリ

### 構成
<pre>
.
├── README.md
├── data
│   ├── AI_E.coli_LS5218_qPCRデータ.xlsx
│   └── qPCR(相対値)_対数増殖期.csv
├── function
│   ├── SEM.py
│   ├── app_sem.py
│   └── sem_plot.py
├── main.py
├── output
│   ├── causalnex
│   │   ├── 4.22_causalnex.png
│   │   ├── 4.22_causalnex_th0.23.png
│   │   ├── 4.22_causalnex_th0.25.png
│   │   ├── 4.22_threshold.png
│   │   ├── 4.23_causalnex_1.png
│   │   └── 4.23_causalnex_2.png
│   ├── output.xlsx
│   └── sem
│       ├── 0.0_semopy
│       ├── 0.0_semopy.png
│       ├── 1.0_semopy
│       ├── 1.0_semopy.png
│       ├── 2.0_semopy
│       ├── 2.0_semopy.png
│       ├── 3.0_semopy
│       ├── 3.0_semopy.png
│       ├── 4.0_semopy
│       └── 4.0_semopy.png
└── work
    ├── app.pkl
    ├── best_sm_2024.pkl
    ├── bio_sample_code.ipynb
    ├── bio_torch_CausalNex_2024.ipynb
    ├── pareto_data_real.csv
    ├── pareto_graph_real.png
    ├── semopy
    ├── semopy.png
    └── test.ipynb
</pre>
