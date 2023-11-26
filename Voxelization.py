from pyOctree import *
import numpy as np
from stl import mesh

# class Voxlization():
#     def __init__(self) -> None:
#         pass
#     def read_stl(file_path):
#     # 读取STL文件
#         mesh_data = mesh.Mesh.from_file(file_path)

#         # 获取三角形数据
#         vertices = mesh_data.vectors.reshape((-1, 3))
        
#         return vertices,mesh_data
#     def boundingBox(triangles):
#         pass



# if __name__ == "__main__":
#     data_path = "B:\Master arbeit\DONUT2.stl"
#     Voxl = Voxlization()
#     triangles,mesh_data = Voxl.read_stl(data_path)
#     # num_triangles = len(mesh_data.vectors)
#     # print("Triangle Vertices:")
#     # print(triangles[:3])  # 打印前三个三角形的顶点坐标
#     print(triangles.shape)
class Voxlization():
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_stl(file_path):
        # 读取STL文件
        mesh_data = mesh.Mesh.from_file(file_path)

        # 获取三角形数据
        vertices = mesh_data.vectors.reshape((-1, 3))
        # triangles = 
        
        return vertices, mesh_data
    @staticmethod
    def boundingBox(triangle):
        min_bound = np.min(triangle,axis=0)
        max_bound = np.max(triangle,axis=0)
        return [min_bound,max_bound]

if __name__ == "__main__":
    data_path = "B:\Master arbeit\DONUT2.stl"
    Voxl = Voxlization()
    triangles, mesh_data = Voxl.read_stl(data_path)
    # num_triangles = len(mesh_data.vectors)
    # print("Triangle Vertices:")
    triangle = triangles[:3]
    print(mesh_data.vectors[0])  # 打印前三个三角形的顶点坐标
    # bounding_box = Voxl.boundingBox(triangle=triangle)
    # print(bounding_box)
    # print(triangles.shape[0]/3)
