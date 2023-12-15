import bpy
import pandas as pd

file_path = "B:\\Master arbeit\\node_data\\inner_nodes7.xlsx"  # 请确保文件路径使用双反斜杠或单斜杠
df = pd.read_excel(file_path)

# 添加原始立方体
bpy.ops.mesh.primitive_cube_add(size=0.078125, location=(0, 0, 0))
cube = bpy.context.active_object

for index, row in df.iterrows():
    x, y, z = row['center_x'], row['center_Y'], row['center_Z']  # 请使用字符串作为列名
    copy_cube = cube.copy()
    copy_cube.location = (x, y, z)  # 修正赋值语句
    bpy.context.collection.objects.link(copy_cube)

bpy.context.view_layer.update()