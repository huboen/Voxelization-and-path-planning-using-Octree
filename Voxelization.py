from pyOctree import *
import numpy as np
from stl import mesh
import cupy as cp
import time

class Voxlization:
    def __init__(self,stl_path) -> None:
        self.stl_path = stl_path

    def read_stl(self,file_path=None):
        if file_path == None:
            file_path = self.stl_path
        
        # read stl file
        mesh_data = mesh.Mesh.from_file(file_path)

        # obtain the triangles
        triangles = mesh_data.vectors
        
        return triangles

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
        bounding_box = [min_bound, max_bound]
        bounding_boxes = cp.stack(bounding_box, axis=-1)
        return bounding_boxes.get()

    @staticmethod
    def boundingBox(triangle):
        min_bound = np.min(triangle, axis=0)
        max_bound = np.max(triangle, axis=0)
        bounding_box = [min_bound, max_bound]
        return bounding_box
    
    def maxBoundingBox(self):
        vertices= self.read_stl(self.stl_path).reshape(-1,3)
        min_bound = np.min(vertices,axis=0)
        max_bound = np.max(vertices,axis=0)
        max_bounding_box = [min_bound,max_bound]
        return max_bounding_box

    

class OctreeOperator(Octree):
    def __init__(self, boundingbox) -> None:
        super().__init__(boundingbox)

    def findBoundingNode(self, target_depth, target_bounding_box, intersected_nodes=None, node=None):
        if intersected_nodes is None:
            intersected_nodes = []

        if node is None:
            node = self.root
        depth = node.depth()
        # Check if the current depth is less than the target depth
        if depth  < target_depth:
            # Check if the bounding boxes intersect
            if self.boundingBoxIntersect(node, target_bounding_box):
                # Extend the node to the next depth
                self.extend(node, depth+1)

                # Recursively search through the node's children
                for child_node in node.children:
                    self.findBoundingNode(target_depth, target_bounding_box, intersected_nodes, child_node)
        else:
            # If the node has reached the target depth, add it to intersected_nodes
            intersected_nodes.append(node)

        return intersected_nodes
    
    def boundingBoxIntersect(self, node, targetBoundingBox):
    # Check if two bounding boxes intersect
        miniNode, maxNode = np.array(node.center) - node.size / 2, np.array(node.center) + node.size / 2
        miniBbox, maxBbox = targetBoundingBox[0], targetBoundingBox[1]

        return not (maxNode[0] < miniBbox[0] or miniNode[0] > maxBbox[0] or
                    maxNode[1] < miniBbox[1] or miniNode[1] > maxBbox[1] or
                    maxNode[2] < miniBbox[2] or miniNode[2] > maxBbox[2] or
                    maxBbox[0] < miniNode[0] or miniBbox[0] > maxNode[0] or
                    maxBbox[1] < miniNode[1] or miniBbox[1] > maxNode[1] or
                    maxBbox[2] < miniNode[2] or miniBbox[2] > maxNode[2])
#array([-0.83045566, -1.2952495 , -0.01996955], dtype=float32) array([-0.8225082 , -1.2088441 ,  0.04101349], dtype=float32)
#array([-4.04194164, -3.79571342, -2.48610242]) array([-1.54194164, -1.29571342,  0.01389758]) 
#array([-4.04194164, -3.79571342,  0.01389758]) array([-1.54194164, -1.29571342,  2.51389758])
#array([-4.04194164, -1.29571342, -2.48610242]) array([-1.54194164,  1.20428658,  0.01389758])
#array([-1.54194164, -3.79571342, -2.48610242]) array([ 0.95805836, -1.29571342,  0.01389758])
#array([-1.54194164, -1.29571342, -2.48610242]) array([0.95805836, 1.20428658, 0.01389758])
if __name__ == "__main__":
    data_path = "B:\Master arbeit\DONUT2.stl"
    voxl = Voxlization(data_path)
    triangles = voxl.read_stl()
    start=time.time()
    bounding_boxes = voxl.boundingBoxes_gpu(triangles)
    end = time.time()
    duration1 = end-start
    start=time.time() 
    bounding_boxes2 = voxl.boundingBoxes(triangles)
    end = time.time()
    duration2 = end-start
    maxBoundingBox = voxl.maxBoundingBox()
    initial_size = np.max(maxBoundingBox[1]-maxBoundingBox[0])
    octreeTest = OctreeOperator(maxBoundingBox)
    targetboundingbox =bounding_boxes[0].T
    
    intersected_nodes = octreeTest.findBoundingNode(target_depth=8,target_bounding_box=targetboundingbox)
    for node in intersected_nodes:
        print(node.depth())
    octreeTest.visualize()
    # print(bounding_boxes[0])
    # print(bounding_boxes2[0])
    # print(len(bounding_boxes))
    # print(len(bounding_boxes2))
    # print("gpu time ")
    # print(duration1)
    # print("cpu time ")
    # print(duration2)
    # print("maxBoundingBox")
    # print(maxBoundingBox)
    # print("size")
    # print(initial_size)
   