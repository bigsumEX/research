import numpy as np
import semopy as sm
from semopy.inspector import inspect


def run_SEM(data, dag_matrix, threshold, target=None):
    '''
    SEMを実行する
    
    Parameters
    ----------
    data: DataFrame
        学習に使用したデータ
        
    dag_matrix: DataFrame
        DAGの接続行列
        
    threshold: float
        エッジの生成確率に対する閾値
        
    target: str
        アウトカム変数（アウトカム変数がある場合、それが含まれる構造のみを残して計算する）
        
    Returns
    -------
    model: obj
        SEMのモデル
        
    fit_index: ndarray(float × 4)
        各種評価指標
        
    inspect: DataFrame
        各エッジのp値などをまとめたもの
    '''

    # DAGマトリクスの値が閾値を超える場合に1、そうでない場合に0の配列を作成
    array_dag_01data = (dag_matrix > threshold)*1
    
    # DAGマトリクスの列名を取得
    columns = dag_matrix.columns

    # 構造方程式を定義
    if target is not None:
        # target（アウトカム変数）が設定されている場合
        # DAGマトリクスを走査し、エッジが存在する場合はエッジモデルをリストに追加
        mod = make_mod(array_dag_01data, detect_subgraph(array_dag_01data, target))
    else:
        mod = make_mod(array_dag_01data)
    
    # 数値の表示オプションを設定
    np.set_printoptions(precision=10, suppress=True)
    
    # データの列名にスペースが含まれている場合はスペースをアンダースコアに置換
    data_bar = data.set_axis([i.replace(' ', '_') for i in data.columns], axis=1)
    # SEMのモデルを作成し、データを読み込む
    model = sm.Model(mod)
    # model.load_dataset(data_bar)
    model.fit(data=data_bar,obj='MLW')
    
    # SEMの最適化を実行し、目的関数の値を取得
    opt = sm.Optimizer(model)
    objective_function_value = opt.optimize()
        
    # SEMの統計情報を収集
    stats = sm.gather_statistics(opt)
    
    # SEMのモデル、評価指標、およびエッジの統計情報を返す        
    return model, np.round([stats.rmsea, stats.gfi, stats.agfi, stats.aic], 5), inspect(opt)

def detect_subgraph(connection_matrix, target):
    """接続行列からアウトカムが含まれるサブグラフのノードを抽出

    Args:
        connection_matrix (pd.DataFrame): 接続行列

    Returns:
        list[str]: アウトカムに関係あるノードのリスト
    """
    # ApgarScoreがつながっているグラフのみを抽出する
    visited = [target]
    queue = []
    # 自身に刺さっているエッジを検出し，基となるノードをリスト化
    queue += connection_matrix[connection_matrix[target] == 1].index.to_list()
    queue += connection_matrix[connection_matrix.loc[target, :] == 1].index.to_list()
    # queueが終わるまで
    while queue:
        # 先頭を取り出す
        visit_node = queue.pop(0)
        visited.append(visit_node)
        
        # print(queue)
        
        # 観測ノードとして取り出したノード
        obs_node = connection_matrix[connection_matrix[visit_node] == 1].index.to_list()
        obs_node += connection_matrix[connection_matrix.loc[visit_node, :] == 1].index.to_list()

        # 検出したノードがvisitedかqueueに含まれないか確認
        obs_node = [node for node in obs_node if node not in visited]
        obs_node = [node for node in obs_node if node not in queue]

        queue += obs_node

    return visited

def make_mod(connection_matrix, nodes=None):
    """接続行列から構造方程式を作成

    Args:
        connection_matrix (pd.DataFrame): 接続行列
        nodes (list[str]): 対象とするノード一覧

    Returns:
        str: 構造方程式
    """
    # 構造方程式を保存するリスト
    structural_equations = []
    
    if nodes is None:
        search_nodes_list = connection_matrix.columns.to_list()
    else:
        search_nodes_list = nodes
    # nodesリストの各変数に対して、親変数を見つけて方程式を作成
    for child in search_nodes_list:
        # 子変数に影響を与える親変数を見つける
        parents = [str(item) for item in connection_matrix.index[connection_matrix[child] == 1].to_list()]
        # 子変数が親変数に影響される方程式を作成
        if parents:
            # 親変数リストの文字列を作成
            parent_str = ' + '.join([i.replace(" ", "_") for i in parents])
            # 方程式をリストに追加
            equation = f"{str(child).replace(' ' , '_')} ~ {parent_str}"
            structural_equations.append(equation)
    # 結果を出力
    return '\n'.join(structural_equations)


def show_sem_grapgh(dag_matrix, data, threshold, file_path):
    """semopyによる因果グラフの描画

    Args:
        dag_matrix (pd.DataFrame): 接続行列
        data (pd.DataFrame): 学習に使用したデータ
        threshold (float): エッジを切るための閾値
        file_path (str): 画像の保存先

    Returns:
        graphviz: graphvizのオブジェクト
        semopy.model: semopyのモデル
    """
    # データの列名にスペースが含まれている場合はスペースをアンダースコアに置換
    # こうしないとsemopyのモデルがエラーを吐く
    new_columns = [i.replace(" ", "_") for i in dag_matrix.columns]
    # 列名の変更
    dag_matrix= dag_matrix.set_axis(new_columns, axis=0)
    dag_matrix= dag_matrix.set_axis(new_columns, axis=1)
    data = data.set_axis(new_columns, axis=1)
    
    # DAGマトリクスの値が閾値を超える場合に1、そうでない場合に0の配列を作成
    array_dag_01data = np.where(dag_matrix > threshold, 1, 0)
    
    # エッジモデルを格納するリストを初期化
    edge_model = []
    
    # DAGマトリクスの列名を取得
    columns = dag_matrix.columns

    # DAGマトリクスを走査し、エッジが存在する場合はエッジモデルをリストに追加
    for i, row in enumerate(array_dag_01data):
        for j, value in enumerate(row):
            if value == 1:
                try:    
                    edge_model.append(columns[j].replace(' ', '_') + " ~ " + columns[i].replace(' ', '_'))
                except:
                    print(array_dag_01data.shape)
                    print(columns)
                    
    # エッジモデルを改行で連結し、モデル文字列を作成
    mod = '\n'.join(edge_model)
    
    model = sm.Model(mod)
    model.fit(data)
    
    # SEMの最適化を実行し、目的関数の値を取得
    opt = sm.Optimizer(model)
    objective_function_value = opt.optimize()
    
    # SEMのプロットを表示
    # node_colors={"Acceleration": "#ec3219"}という引数をつけると、特定のノードの色と接続するエッジの色を変更できる
    graph = sm.semplot(model, file_path, inspection=inspect(opt),
                        plot_ests=True, engine='dot',
                       fontsize_node=20, fontsize_edge=14, arrowsize=1.5)
        
    return graph, model