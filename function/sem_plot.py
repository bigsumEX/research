import pandas as pd
from function.SEM import run_SEM
from openpyxl import Workbook
from openpyxl.drawing.image import Image

def sem_plot_to_excel(obj, num=10 , excel_file="./output/output.xlsx"):

    df = obj.study.trials_dataframe()
    df_top = df.sort_values("value").head(num)
    # ソートされた順番に応じてnumber列を書き換える
    df_top["number"] = range(0, num)

    df_top = df_top[["number", "value", "params_lasso_beta","params_ridge_beta", "params_threshold"]]
    df_top.columns = ["number", "RMSEA", "lasso_beta", "ridge_beta", "threshold"]
    
    # Excelファイルを作成
    wb = Workbook()
    ws = wb.active
    ws.title = "Top Results"

    # カラム名を書き込む
    ws.append(list(df_top.columns))

    # DataFrameをExcelに書き込む
    for r_idx, row in enumerate(df_top.itertuples(), start=2):
        for c_idx, value in enumerate(row[1:], start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # SEMのグラフを挿入
    for r_idx, row in enumerate(df_top.itertuples(), start=1):
        # sm_SEM = run_SEM(obj.df, obj.matrix_list[int(row.number)], row.threshold)
        sem_graph_path = f"./output/sem/{float(row.number)}_semopy.png"
        
        # 新しいシートを作成し、各行のデータフレームを書き込む
        ws_new = wb.create_sheet(title=f"SEM Result {float(row.number)}")
        ws_new.append(["number", "RMSEA", "lasso_beta", "ridge_beta", "threshold"])
        ws_new.append([getattr(row, field) for field in df_top.columns])
        
        # SEMのグラフを挿入
        img = Image(sem_graph_path)
        img.anchor = ws_new.cell(row=3, column=1).coordinate
        ws_new.add_image(img)

    # Excelファイルを保存
    wb.save(excel_file)