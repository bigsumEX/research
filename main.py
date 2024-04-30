from function.app_sem import APP_SEM
from function.sem_plot import sem_plot
import pickle

def main():
    app = APP_SEM()
    app.fit(n_trials=10)
    # app.pareto_trial(n_trials=100)

    # モデルを保存
    with open('./output/pkl/app.pkl_from_pandas_lasso_agfi', 'wb') as file:
        pickle.dump(app, file)

    # 保存したモデルを読み込む
    with open('./output/pkl/app.pkl_from_pandas_lasso_agfi', 'rb') as file:
        loaded_app = pickle.load(file)

    sem_plot(loaded_app, num=3, filename='./output/excel/from_pandas_lasso_agfi.xlsx')

if __name__ == "__main__":
    main()
    