from function.app_sem import APP_SEM
from function.sem_plot import sem_plot_to_excel
import pickle

def main():
    app = APP_SEM()
    app.fit(n_trials=5)

    # モデルを保存
    with open('./output/pkl/app.pkl', 'wb') as file:
        pickle.dump(app, file)

    # 保存したモデルを読み込む
    with open('./output/pkl/app.pkl', 'rb') as file:
        loaded_app = pickle.load(file)

    sem_plot_to_excel(loaded_app, num=5)

if __name__ == "__main__":
    main()
    