import pandas as pd
from function.SEM import run_SEM
from openpyxl import Workbook
from openpyxl.drawing.image import Image

def sem_plot_to_excel(obj, num=10 , excel_file="./output/output.xlsx"):

    df = obj.study.trials_dataframe()
    df = df[["number", "value", "params_lasso_beta","params_ridge_beta", "params_threshold"]]
    df.columns = ["number", "value", "lasso_beta", "ridge_beta", "threshold"]

    df = pd.concat([df, obj.df_stats.reset_index(drop=True)], axis=1)
    #昇順の場合：asceding=False
    df_top = df.sort_values("value").head(num)
    # ソートされた順番に応じてnumber列を書き換える
    df_top["number"] = range(0, num)
    df_result = df_top.loc[:, :"threshold"]
    df_stats = df_top.loc[:, "DoF":]

    # Excelファイルを作成
    wb = Workbook()
    ws = wb.active
    ws.title = "Top Results"

    # カラム名を書き込む
    ws.append(list(df_result.columns))
    # DataFrameをExcelに書き込む
    for r_idx, row in enumerate(df_result.itertuples(), start=2):
        for c_idx, value in enumerate(row[1:], start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # カラム名を書き込む
    ws.append(list(df_stats.columns))
    # DataFrameをExcelに書き込む
    for r_idx, row in enumerate(df_stats.itertuples(), start=len(df_top) + 4):
        for c_idx, value in enumerate(row[1:], start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # SEMのグラフを挿入
    for index, row in df_top.iterrows():
        sem_graph_path = f"./output/sem/{float(row.number)}_semopy.png"
        
        # 新しいシートを作成し、各行のデータフレームを書き込む
        ws_new = wb.create_sheet(title=f"SEM Result {float(row.number)}")
        ws_new.append(list(df_top.columns))
        row_data = [row[col] for col in df_top.columns]
        ws_new.append(row_data) 
        # SEMのグラフを挿入
        img = Image(sem_graph_path)
        img.anchor = ws_new.cell(row=3, column=1).coordinate
        ws_new.add_image(img)

    # Excelファイルを保存
    wb.save(excel_file)