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
❯ tree /f
</pre>
