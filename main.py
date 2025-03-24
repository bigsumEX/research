from function.app_sem import APP_SEM
from function.sem_plot import sem_plot
from function.pareto_sem_plot import pareto_sem_plot
import pickle

def main():
    app = APP_SEM()
    # app.fit(n_trials=100)
    app.pareto_trial(n_trials=10)

    # モデルを保存
    with open('./output/pkl/test_02_12', 'wb') as file:
        pickle.dump(app, file)

    # 保存したモデルを読み込む
    with open('./output/pkl/test_02_12', 'rb') as file:
        loaded_app = pickle.load(file)

    # # sem_plot(loaded_app, num=10, filename='./output/excel/pytorch_from_pandas_parato_rmsea_agfi.xlsx')
    # pareto_sem_plot(loaded_app, num=20, filename='./output/excel/pytorch_from_pandas_pareto_rmsea_agfi_zousyoku_th0.2_0.6_0109_trial300.xlsx')
if __name__ == "__main__":
    main()
    