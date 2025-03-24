import pandas as pd
import numpy as np

from causalnex.structure.pytorch import from_pandas
from causalnex.structure import StructureModel
import optuna
from optuna.samplers import MOTPESampler
from optuna.exceptions import TrialPruned

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from function.SEM import run_SEM
import traceback


class APP_SEM:
    def __init__(self):
        self.data_load()
        self.sm_list = []  # 辞書に変更
        self.matrix_dict = {}  # 辞書に変更
        self.df_stats = pd.DataFrame()


    def objective(self, trial):
        # Optunaでチューニングするハイパーパラメータ
        # beta = trial.suggest_float('beta', 1e-2, 1e-1)
        lasso_beta = trial.suggest_float('lasso_beta', 1e-2, 1e-1,  log=True)
        ridge_beta = trial.suggest_float('ridge_beta', 1e-2, 1e-1, log=True)
        threshold = trial.suggest_float('threshold', 0.2, 0.6)

        # StructureModelのインスタンスを作成
        sm = StructureModel()

        # NOTEARSアルゴリズムを用いて構造学習を実施
        # ここでfrom_pandaslassoのパラメータbetaをOptunaのtrialを通してチューニング
        sm = from_pandas(
            self.df,
            # beta=beta
            lasso_beta=lasso_beta,
            ridge_beta = ridge_beta
        )

        sm.remove_edges_below_threshold(threshold)

        try:
            # 構造をDAGに修正し、最大の部分グラフを取得
            sm.threshold_till_dag()
            sm_l = sm.get_largest_subgraph()

            # smを隣接行列に変換
            connection_matrix = pd.DataFrame(self.sm_to_dag_matrix(sm_l))
            
            #既存の行列と比較して同一であるか確認
            for trial_number, existing_matrix in self.matrix_dict.items():
                if connection_matrix.equals(existing_matrix):
                    raise TrialPruned("既存の行列と重複したため、Trialを回避します")
            
            self.matrix_dict[trial.number] = connection_matrix  # 辞書に追加

            print("隣接行列")
            print("="*20)
            print(connection_matrix)

            try:
                 _, stats, _ = run_SEM(self.df, connection_matrix, threshold)
            except:
                trial.set_user_attr('error', True)
                raise TrialPruned("run_SEM実行時に異常を検知しました。Trialを回避します")

            # 指標のチェックとエラーハンドリング
            if stats["RMSEA"]["Value"] == 0.0 or stats["GFI"]["Value"] == 1.0 or stats["AGFI"]["Value"] == 1.0 or stats["AIC"]["Value"] == 0.0:
                raise optuna.TrialPruned("評価指標に異常を検知しました。Trialを回避します")

            elif stats["RMSEA"]["Value"] >= 0.15 or stats["GFI"]["Value"] <= 0.85 or stats["AGFI"]["Value"] <= 0.85 or stats["AIC"]["Value"] >= 100:
                raise optuna.TrialPruned("評価指標が悪いため、Trialを回避します")


            # 成功した場合の処理
            
            self.df_stats = pd.concat([self.df_stats, stats])
            trial.set_user_attr('best_sm', sm_l)
            self.sm_list.append(sm_l)

            return stats["RMSEA"]["Value"], stats["AGFI"]["Value"]

        except optuna.TrialPruned:
            print(traceback.format_exc())
            trial.set_user_attr('error', True)
            return float(1e10), -float(1e10)  # Noneを避けるため、ペナルティ値を返す
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            trial.set_user_attr('error', True)
            # 高いペナルティ値で返す
            return float(1e10), -float(1e10)  # Noneを避けるため、ペナルティ値を返す

    def sm_to_dag_matrix(self, sm: StructureModel):
        """smの因果グラフを接続行列に変換する

        Args:
            sm (StructureModel): smの因果グラフ

        Returns:
            np.array: 接続行列
        """
        # 因果グラフのノード数
        try:
            nodes = sm.nodes
        except:
            return np.array([])
            
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
    
    def data_load(self, path = './data/qPCR(相対値)_定常期.csv', scaler=True):
        df = pd.read_csv(path, header=0)
        df = df.dropna()
        df = df.drop(['gene', '培養時間'], axis=1)
        df = df.reset_index(drop=True)
        if scaler:
            # scaler = MinMaxScaler()
            scaler = StandardScaler()
            normalized_data_array = scaler.fit_transform(df)
            # DataFrame型に変換
            df = pd.DataFrame(normalized_data_array, columns=df.columns)

        self.df = df

    def fit(self, n_trials=100):
        # スコア(エッジの数)を最大化するように設定
        #RMSEAの場合:minimize, #GFI,AGFIの場合:maximize
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective, n_trials=n_trials, catch=(TrialPruned, Exception))
        # ログ非表示
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def pareto_trial(self, n_trials=100):
        self.study = optuna.create_study(
            directions=['minimize', 'maximize'],
            sampler=MOTPESampler()
            )
    
        self.study.optimize(self.objective, n_trials=n_trials, catch=(TrialPruned, Exception))
        # ログ非表示
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print("Number of finished trials:", len(self.study.trials))

        print("Pareto front")
        for trial in self.study.best_trials:
            print(f" Values: {trial.values}, Params: {trial.params}")
