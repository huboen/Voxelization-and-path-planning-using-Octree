import bpy
import pandas as pd

file_path3 = "B:\\Master arbeit\\node_data\\inner_nodes9.xlsx"
sheet_name = "Sheet_1"

df3 = pd.read_excel(file_path3, sheet_name=sheet_name)

# 清除场景中的所有网格对象
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# 创建材质1 - 红色
material1 = bpy.data.materials.new(name="MyMaterial1")
material1.use_nodes = False  # 关闭节点编辑器
material1.diffuse_color = (1, 0, 0, 1)  # 这里使用 RGBA 表示红色

# 创建立方体模板1
bpy.ops.mesh.primitive_cube_add(size=df3["size"][0], location=(0, 0, 0))
template_cube1 = bpy.context.active_object
template_cube1.data.materials.append(material1)

# 根据 Excel 数据创建对象1
for index, row in df3.iterrows():
    if row["depth"] == 4:
        x, y, z = row['center_x'], row['center_Y'], row['center_Z']
        
        # 创建新立方体对象1
        new_cube1 = bpy.data.objects.new("Cube1", template_cube1.data.copy())
        new_cube1.location = (x, y, z)
        
        # 将对象链接到场景中的集合
        bpy.context.collection.objects.link(new_cube1)

# 删除模板立方体1
bpy.data.objects.remove(template_cube1, do_unlink=True)

# 创建材质2 - 蓝色
material2 = bpy.data.materials.new(name="MyMaterial2")
material2.use_nodes = False  # 关闭节点编辑器
material2.diffuse_color = (0, 0, 1, 1)  # 这里使用 RGBA 表示蓝色

# 创建立方体模板2
bpy.ops.mesh.primitive_cube_add(size=df3["size"][70], location=(0, 0, 0))
template_cube2 = bpy.context.active_object
template_cube2.data.materials.append(material2)

# 根据 Excel 数据创建对象2
for index, row in df3.iterrows():
    if row["depth"] == 5:
        x, y, z = row['center_x'], row['center_Y'], row['center_Z']
        
        # 创建新立方体对象2
        new_cube2 = bpy.data.objects.new("Cube2", template_cube2.data.copy())
        new_cube2.location = (x, y, z)
        
        # 将对象链接到场景中的集合
        bpy.context.collection.objects.link(new_cube2)

# 删除模板立方体2
bpy.data.objects.remove(template_cube2, do_unlink=True)

# 创建材质3 - 绿色
material3 = bpy.data.materials.new(name="MyMaterial3")
material3.use_nodes = False  # 关闭节点编辑器
material3.diffuse_color = (0, 1, 0, 0.25)  # 这里使用 RGBA 表示绿色

# 创建立方体模板3
bpy.ops.mesh.primitive_cube_add(size=df3["size"][3231], location=(0, 0, 0))
template_cube3 = bpy.context.active_object
template_cube3.data.materials.append(material3)

# 根据 Excel 数据创建对象3
for index, row in df3.iterrows():
    if row["depth"] == 6:
        x, y, z = row['center_x'], row['center_Y'], row['center_Z']
        
        # 创建新立方体对象3
        new_cube3 = bpy.data.objects.new("Cube3", template_cube3.data.copy())
        new_cube3.location = (x, y, z)
        
        # 将对象链接到场景中的集合
        bpy.context.collection.objects.link(new_cube3)

# 删除模板立方体3
bpy.data.objects.remove(template_cube3, do_unlink=True)

# 选择所有对象
bpy.ops.object.select_all(action='SELECT')

# 如果有选中的对象，则设置最后一个选中的对象为活动对象
if bpy.context.selected_objects:
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[-1]
    
    # 检查是否至少有两个选中的对象，以避免合并失败
    if len(bpy.context.selected_objects) >= 2:
        bpy.ops.object.join()
    else:
        print("至少需要两个选中的对象才能执行合并操作。")
else:
    print("没有选中的对象，无法执行合并操作。")
