from pyOctree import *
import numpy as np
from stl import mesh

from stl import mesh
import numpy as np

class Voxlization:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_stl(file_path):
        # 读取 STL 文件
        mesh_data = mesh.Mesh.from_file(file_path)

        # 获取三角形数据
        vertices = mesh_data.vectors.reshape((-1, 3))
        
        return vertices

    @staticmethod
    def boundingBoxes(triangles):
        boundingBoxGroup = []
        for triangle in triangles:
            boundingBoxGroup.append(Voxlization.boundingBox(triangle))
        return boundingBoxGroup

    @staticmethod
    def boundingBox(triangle):
        min_bound = np.min(triangle, axis=0)
        max_bound = np.max(triangle, axis=0)
        return [min_bound, max_bound]

if __name__ == "__main__":
    data_path = "B:\Master arbeit\DONUT2.stl"
    voxl = Voxlization()
    triangles = voxl.read_stl(data_path)
    bounding_boxes = voxl.boundingBoxes(triangles)
    print(bounding_boxes)