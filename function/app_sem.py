import pandas as pd
import numpy as np

from causalnex.structure.pytorch import from_pandas
from causalnex.structure import StructureModel
import optuna

from sklearn.preprocessing import MinMaxScaler
from function.SEM import run_SEM


class APP_SEM:
    def __init__(self):
        self.data_load()
        self.sm_list = []
        self.matrix_list = []

    
    def objective(self, trial):
        # Optunaでチューニングするハイパーパラメータ
        threshold = trial.suggest_float('threshold', 0.2, 0.4)
        lasso_beta = trial.suggest_float('lasso_beta', 1e-2, 1e-1, log=True)  # ログスケールでlassoの値を探索
        ridge_beta = trial.suggest_float('ridge_beta', 1e-2, 1e-1, log=True)  # リッジ正則化の係数を探索

        # StructureModelのインスタンスを作成
        sm = StructureModel()

        # NOTEARSアルゴリズムを用いて構造学習を実施
        # ここでfrom_pandasのパラメータをOptunaのtrialを通してチューニング
        sm = from_pandas(self.df, 
                        lasso_beta=lasso_beta,
                        ridge_beta=ridge_beta,
                        )
        
        #from_pandasで学習した後に閾値を探索する
        sm.remove_edges_below_threshold(threshold)
        #構造をDAG構造に修正
        sm.threshold_till_dag()
        sm_l = sm.get_largest_subgraph()
        self.sm_list.append(sm_l)

        # view_graph_from_sm(sm, "a", False, True)
        #smを接続行列に変換
        connection_matrix = pd.DataFrame(self.sm_to_dag_matrix(sm_l))
        self.matrix_list.append(connection_matrix)

        #run_SEMを実行
        try:
            sm_SEM = run_SEM(self.df, connection_matrix, threshold)
            # 学習された構造のスコアを計算（スコアリング方法はプロジェクトにより異なる）
            score = sm_SEM[1]
            rmsea = score[0]
            # aic = score[3]

            # rmseaが0.0の場合も2.0に設定
            if rmsea == 0.0:
                rmsea = 2.0
                
        except Exception as e:
            #SEM学習時にエラーを吐かれた場合は、rmsea値を2.0としエラー回避する。
            rmsea = 2.0
            # aic = 1000
        
        trial.set_user_attr('best_sm', sm)
        return rmsea
    
    def sm_to_dag_matrix(self, sm: StructureModel):
        """smの因果グラフを接続行列に変換する

        Args:
            sm (StructureModel): smの因果グラフ

        Returns:
            np.array: 接続行列
        """
        # 因果グラフのノード数
        nodes = sm.nodes
        n = len(nodes)

        # ノード名をインデックスに変換する辞書
        node_index = {node: i for i, node in enumerate(nodes)}

        # 接続行列の初期化
        connection_matrix = np.zeros((n, n))

        # 接続行列の作成
        for from_node, to_node in sm.edges:
            connection_matrix[node_index[from_node], node_index[to_node]] = 1
            
        # データフレームに変換して、特徴量名の設定する
        feature_names = list(node_index.keys())
        df_connection_matrix = pd.DataFrame(connection_matrix, columns=feature_names, index=feature_names)

        return df_connection_matrix
    
    def data_load(self, path = './data/qPCR(相対値)_対数増殖期.csv', scaler=True):
        df = pd.read_csv(path, header=0)
        df = df.dropna()
        df = df.drop(['gene', '培養時間'], axis=1)
        df = df.reset_index(drop=True)
        if scaler:
            scaler = MinMaxScaler()
            normalized_data_array = scaler.fit_transform(df)
            # DataFrame型に変換
            df = pd.DataFrame(normalized_data_array, columns=df.columns)

        self.df = df

    def fit(self, n_trials=100):
        # スコア(エッジの数)を最大化するように設定
        self.study = optuna.create_study(direction='minimize')
        # study = optuna.multi_objective.create_study(directions=['minimize', 'minimize'])
        # 100回の試行で最適化
        self.study.optimize(self.objective, n_trials)
        # ログ非表示
        optuna.logging.set_verbosity(optuna.logging.WARNING)