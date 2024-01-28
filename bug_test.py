import pandas as pd
import os

files_with_changes = []
folder_path = "B:\\Master arbeit\\layer_data"
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        file_path = os.path.join(folder_path, file_name)

        # 读取 Excel 文件
        df = pd.read_excel(file_path,sheet_name="Sheet_1")
        print(file_path)

        # 检查是否存在 depth 列
        if 'depth' in df.columns:
            # 检查 depth 列是否有变化
            if df["depth"][0] == 8:
            # if len(unique_depth_values) > 1:
                # 发现变化，将文件名添加到列表中
                files_with_changes.append(file_name)
                os.remove(file_path)
