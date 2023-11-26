from pyOctree import *
import numpy as np
from stl import mesh
import cupy as cp
import time

class Voxlization:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_stl(file_path):
        # 读取 STL 文件
        mesh_data = mesh.Mesh.from_file(file_path)

        # 获取三角形数据
        vertices = mesh_data.vectors
        
        return vertices

    @staticmethod
    def boundingBoxes(triangles):
        boundingBoxGroup = []
        for triangle in triangles:
            boundingBoxGroup.append(Voxlization.boundingBox(triangle))
        return boundingBoxGroup
    
    @staticmethod
    def boundingBoxes_gpu(triangles):
        triangles_gpu = cp.asarray(triangles)
        min_bound = cp.min(triangles_gpu, axis=1)
        max_bound = cp.max(triangles_gpu, axis=1)
        bounding_boxes = cp.stack([min_bound, max_bound], axis=-1)
        return bounding_boxes.get()

    @staticmethod
    def boundingBox(triangle):
        min_bound = np.min(triangle, axis=0)
        max_bound = np.max(triangle, axis=0)
        return [min_bound, max_bound]
    

class OctreeOperator(Octree):
    def __init__(self, boundingbox) -> None:
        super().__init__(boundingbox)

    def findBoundingNode(self, target_depth, target_bounding_box, intersected_nodes=None, node=None):
        if intersected_nodes is None:
            intersected_nodes = []

        if node is None:
            node = self.root

        # Check if the current depth is less than the target depth
        if node.depth() < target_depth:
            # Check if the bounding boxes intersect
            if self.boundingBoxIntersect(node, target_bounding_box):
                # Extend the node to the next depth
                self.extend(node, 1)

                # Recursively search through the node's children
                for child_node in node.children:
                    self.findBoundingNode(target_depth, target_bounding_box, intersected_nodes, child_node)
        else:
            # If the node has reached the target depth, add it to intersected_nodes
            intersected_nodes.append(node)

        return intersected_nodes


if __name__ == "__main__":
    data_path = "B:\Master arbeit\DONUT2.stl"
    voxl = Voxlization()
    triangles = voxl.read_stl(data_path)
    start=time.time()
    bounding_boxes = voxl.boundingBoxes_gpu(triangles)
    end = time.time()
    duration1 = end-start
    start=time.time() 
    bounding_boxes2 = voxl.boundingBoxes(triangles)
    end = time.time()
    duration2 = end-start
    print(bounding_boxes[0])
    print(bounding_boxes2[0])
    print(len(bounding_boxes))
    print(len(bounding_boxes2))
    print("gpu time ")
    print(duration1)
    print("cpu time ")
    print(duration2)