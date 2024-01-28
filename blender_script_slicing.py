import bpy
import pandas as pd

file_path1 = "B:\\Master arbeit\\layer_data\\layer-0.1325867921113968.xlsx"

sheet_name = "Sheet_1"

df1 = pd.read_excel(file_path1, sheet_name=sheet_name)


# 清除场景中的所有网格对象
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# 创建材质1 - 红色
material1 = bpy.data.materials.new(name="MyMaterial1")
material1.use_nodes = False  # 关闭节点编辑器
material1.diffuse_color = (1, 0, 0, 1)  # 这里使用 RGBA 表示红色

# 创建立方体模板1
bpy.ops.mesh.primitive_cube_add(size=df1.loc[0, 'size'], location=(0, 0, 0))
template_cube1 = bpy.context.active_object
template_cube1.data.materials.append(material1)

# 根据 Excel 数据创建对象1
for index, row in df1.iterrows():
    x, y, z = row['center_x'], row['center_Y'], row['center_Z']
    
    # 创建新立方体对象1
    new_cube1 = bpy.data.objects.new("Cube1", template_cube1.data.copy())
    new_cube1.location = (x, y, z)
    
    # 将对象链接到场景中的集合
    bpy.context.collection.objects.link(new_cube1)

bpy.data.objects.remove(template_cube1, do_unlink=True)

# 合并所有立方体
bpy.ops.object.select_all(action='SELECT')

# 如果有选中的对象，则设置最后一个选中的对象为活动对象
if bpy.context.selected_objects:
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[-1]
    bpy.ops.object.join()
else:
    print("没有选中的对象，无法执行合并操作。")
