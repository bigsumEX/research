# NOTEARSとSEMによる構造学習
<div id="top"></div>

## 使用技術一覧

<!-- シールド一覧 -->
<p style="display: inline">
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
</p>

## 分析の流れ
2. データ前処理
3. Optunaによるハイパーパラメータのベイズ最適化, 構造学習
4. ハイパーパラメータの選定, DAGの描画
 
## 分析方法の詳細
### データ前処理
- データの正規化
    - モチベーション: データの大小が極端なので足並みを揃える
    - 手法: MinMaxScalerによる0~1範囲での正規化

### Optunaによるハイパーパラメータのベイズ最適化, 構造学習
- pytorch.from_padasの学習に用いるパラメータを最適化する
- 最適化の評価指標: RMSEA, AGFI
    - RMSEA, AGFI: 共分散構造分析(Semopy)によるモデルとデータの適合度を表現する評価指標
- 探索するパラメータ: lasso_beta, ridge_beta, threshold

### ハイパーパラメータの選定, DAGの描画
- RMSEAを最小化, AGFIを最大化するように構造を学習する
- RMSEAは0.15以上切り捨て, AGFIは0.9以下切り捨てで上位20個がexcelに自動的にまとめられる
- kakunin.ipynbではさらに精度が良好だった各モデルのエッジを再現率として表形式, ヒートマップでまとめた

## 使い方
1. ./function/app_sem.py
    1. objective関数でパラメータ探索の範囲を決める
    2. data_load関数で分析するデータを指定する

2. ./main.py
    1. 単目的最適化を行う場合はfit関数を使用、多目的最適化を行う場合はpareto_trial関数を使用
    2. pklとして保存する
    3. 保存したpklを読み込む
    4. 単目的最適化の場合はsem_plot関数、多目的最適化の場合はpareto_sem_plot関数を使用してexcelにネットワーク, パラメータ, 評価指標を自動保存する

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
│   ├── AI_E.coli_LS5218_qPCRデータ.xlsx
│   ├── qPCR(相対値)_定常期.csv
│   ├── qPCR(相対値)_対数増殖期+定常期.csv
│   └── qPCR(相対値)_対数増殖期.csv
├── function
│   ├── SEM.py
│   ├── app_sem.py
│   ├── pareto_sem_plot.py
│   ├── plot.py
│   └── sem_plot.py
├── kakunin.ipynb
├── main.py
├── output
│   ├── excel
│   ├── pkl
│   ├── png
│   └── sem
└── tree.txt
</pre>
