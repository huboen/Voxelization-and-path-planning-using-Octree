from pyOctree import *
import numpy as np
from stl import mesh
import cupy as cp
import time
import cProfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from numba import cuda

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
    #recursion
    def findBoundingNode_recursion(self, target_depth, target_bounding_box, intersected_nodes=None, node=None):
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
                    self.findBoundingNode_recursion(target_depth, target_bounding_box, intersected_nodes, child_node)
        else:
            # If the node has reached the target depth, add it to intersected_nodes
            intersected_nodes.append(node)

        return intersected_nodes


    #iteration
    def findBoundingNode_iteration(self, target_depth, target_bounding_box):
        intersected_nodes = []
        stack = [(self.root, 0)]  # 初始栈，包含根节点和深度 0

        while stack:
            node, depth = stack.pop()

            # Check if the current depth is less than the target depth
            if depth < target_depth:
                # Check if the bounding boxes intersect
                if self.boundingBoxIntersect(node, target_bounding_box):
                    # Extend the node to the next depth
                    self.extend(node, depth + 1)

                    # Add the children to the stack for further exploration
                    for child_node in node.children:
                        stack.append((child_node, depth + 1))
            else:
                # If the node has reached the target depth, add it to intersected_nodes
                intersected_nodes.append(node)

        return intersected_nodes
    def boundingBoxIntersect(self, node, targetBoundingBox):
    # Check if two bounding boxes intersect
        miniNode, maxNode = np.array(node.center) - node.size / 2, np.array(node.center) + node.size / 2 
        miniBbox, maxBbox= targetBoundingBox[0], targetBoundingBox[1]

        return not (maxNode[0] < miniBbox[0] or miniNode[0] > maxBbox[0] or
                    maxNode[1] < miniBbox[1] or miniNode[1] > maxBbox[1] or
                    maxNode[2] < miniBbox[2] or miniNode[2] > maxBbox[2] or
                    maxBbox[0] < miniNode[0] or miniBbox[0] > maxNode[0] or
                    maxBbox[1] < miniNode[1] or miniBbox[1] > maxNode[1] or
                    maxBbox[2] < miniNode[2] or miniBbox[2] > maxNode[2])
    @staticmethod
    @cuda.jit
    def boundingBoxIntersect_cuda(centers, sizes, targetBoundingBoxes, results):
        i = cuda.grid(1)
        if i < len(centers):
            # miniNode, maxNode = centers[i] - sizes[i] / 2, centers[i] + sizes[i] / 2 
            # miniBbox, maxBbox= targetBoundingBoxes[0], targetBoundingBoxes[1]
                miniNode = cuda.device_array(3, dtype=numba.float64)
                miniNode[0] = centers[i, 0] - sizes[i, 0]/2
                miniNode[1] = centers[i, 1] - sizes[i, 1]/2
                miniNode[2] = centers[i, 2] - sizes[i, 2]/2

        # results[i] = not (maxNode[0].astype(cp.int64) < miniBbox[0] or miniNode[0].astype(cp.int64) > maxBbox[0] or
        #           maxNode[1].item().astype(cp.int64) < miniBbox[1] or miniNode[1].astype(cp.int64) > maxBbox[1] or
        #           maxNode[2].item().astype(cp.int64) < miniBbox[2] or miniNode[2].astype(cp.int64) > maxBbox[2] or
        #           maxBbox[0] < miniNode[0].item().astype(cp.int64) or miniBbox[0] > maxNode[0].astype(cp.int64) or
        #           maxBbox[1] < miniNode[1].item().astype(cp.int64) or miniBbox[1] > maxNode[1].astype(cp.int64) or
        #           maxBbox[2] < miniNode[2].item().astype(cp.int64) or miniBbox[2] > maxNode[2].astype(cp.int64))
            
    def findBoundingNode_cuda(self, target_depth, target_bounding_box):
        centers, sizes = OctreeOperator.maxtrixOperator(self.all_leaf_nodes())
        print(centers.shape)
        print(sizes.shape)
        # 将数据传递到 GPU
        # 将数据传递到 GPU
        centers_gpu = cuda.to_device(np.array(centers))
        sizes_gpu = cuda.to_device(np.array(sizes))
        target_bounding_box_gpu = cuda.to_device(np.array(target_bounding_box))
        result_gpu = cuda.device_array((centers_gpu.shape[0],centers_gpu.shape[1]),dtype=bool)

        # 设置线程和块大小
        threads_per_block = 256
        blocks_per_grid = (centers_gpu.shape[0] + threads_per_block - 1) // threads_per_block

        # 调用 CUDA 核函数
        OctreeOperator.boundingBoxIntersect_cuda[blocks_per_grid, threads_per_block](centers_gpu, sizes_gpu, target_bounding_box_gpu, result_gpu)

        # 从 GPU 获取结果
        result = result_gpu.copy_to_host()
        return result
    
    def __update_octree__(self, results, target_depth):
        for label, node in enumerate(results,self.leafnodes):
            depth = node.depth()
            if label and depth < target_depth:
                self.extend(node, depth+1)
        self.all_leaf_nodes()
    @staticmethod
    def maxtrixOperator(nodes):
        centers = []
        sizes = []
        for node in nodes:
            centers.append(node.center)
            sizes.append(np.full(3, node.size, dtype=np.float64))
        return np.array(centers), np.array(sizes)
if __name__ == "__main__":
    data_path = "B:\Master arbeit\DONUT2.stl"
    voxl = Voxlization(data_path)
    triangles = voxl.read_stl()
    # start=time.time()
    bounding_boxes = voxl.boundingBoxes_gpu(triangles)
    # end = time.time()
    # duration1 = end-start
    # start=time.time() 
    # bounding_boxes2 = voxl.boundingBoxes(triangles)
    # end = time.time()
    # duration2 = end-start
    maxBoundingBox = voxl.maxBoundingBox()
    # initial_size = np.max(maxBoundingBox[1]-maxBoundingBox[0])
    # start=time.time() 
    octreeTest = OctreeOperator(maxBoundingBox)
    # end = time.time()
    # initial_duration = end-start
    targetboundingbox =bounding_boxes[0].T
    # position = octreeTest.root.center
    # start=time.time() 
    # intersected_nodes = octreeTest.findBoundingNode_recursion(target_depth=10,target_bounding_box=targetboundingbox)
    # end = time.time()
    # duration3 = end-start
    # octreeTest2 = OctreeOperator(maxBoundingBox)
    # start=time.time() 
    # intersected_nodes = octreeTest2.findBoundingNode_iteration(target_depth=10,target_bounding_box=targetboundingbox)
    # end = time.time()
    # duration4 = end-start
    # # for node in intersected_nodes:
    # #     print(node.depth())
    # octreeTest.visualize()
    # # print(bounding_boxes[0])
    # # print(bounding_boxes2[0])
    # print(len(bounding_boxes))
    # print(len(bounding_boxes2))
    # print("initial_duration")
    # print(initial_duration)
    # print("gpu time ")
    # print(duration1)
    # print("cpu time ")
    # print(duration2)
    # print("maxBoundingBox")
    # print(maxBoundingBox)
    # print("size")
    # print(initial_size)
    # print("number of intersected nodes")
    # print(len(intersected_nodes))
    # print("findingtime recursive")
    # print(duration3)
    # print("findingtime iterative")
    # print(duration4)
#     profiler = cProfile.Profile()

# # 开始性能分析
#     profiler.enable()

    # 运行你的函数
    # intersected_nodes = octreeTest.findBoundingNode_recursion(target_depth=10, target_bounding_box=targetboundingbox)
    results = octreeTest.findBoundingNode_cuda(target_depth=10, target_bounding_box=targetboundingbox)
    print(results)
    # octreeTest.all_leaf_nodes()   

    # 结束性能分析
    # profiler.disable()

    # # 生成性能报告
    # profiler.print_stats(sort='cumulative')  # 按照累积时间排序