from pyOctree import *
import numpy as np
from stl import mesh
import cupy as cp
import time
import cProfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from numba import cuda
import numba

class Voxlization:
    def __init__(self,stl_path) -> None:
        self.stl_path = stl_path

    def read_stl(self,file_path=None):
        if file_path == None:
            file_path = self.stl_path
        
        # read stl file
        # mesh_data = o3d.io.read_triangle_mesh(file_path)
        mesh_data = mesh.Mesh.from_file(file_path)
        # obtain the triangles
        # triangles = mesh_data.triangles
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
    def boundingBoxIntersect_cuda(centers, sizes, targetBoundingBoxes, results,output_test = None):
        thread= cuda.grid(1)
        if thread < targetBoundingBoxes.shape[0]:
            # miniNode, maxNode = centers[i] - sizes[i] / 2, centers[i] + sizes[i] / 2 
                # 先获取 targetBoundingBoxes[i] 切片
                if output_test is not None:
                    output_test[thread] = thread
                bbox_slice = targetBoundingBoxes[thread]
                
                # 然后再获取 [0, :] 切片
                miniBbox = cuda.local.array(3, dtype=numba.float64)
                maxBbox = cuda.local.array(3, dtype=numba.float64)
                miniBbox, maxBbox = bbox_slice[0], bbox_slice[1]
                for node_number in range(len(centers)):
                    miniNode = cuda.local.array(3, dtype=numba.float64)
                    maxNode = cuda.local.array(3, dtype=numba.float64)
                    miniNode[0] = centers[node_number, 0] - sizes[node_number, 0]/2
                    miniNode[1] = centers[node_number, 1] - sizes[node_number, 1]/2
                    miniNode[2] = centers[node_number, 2] - sizes[node_number, 2]/2
                    maxNode[0] = centers[node_number, 0] + sizes[node_number, 0]/2
                    maxNode[1] = centers[node_number, 1] + sizes[node_number, 1]/2
                    maxNode[2] = centers[node_number, 2] + sizes[node_number, 2]/2

                    # if not (maxNode[0] < miniBbox[0] or miniNode[0] > maxBbox[0] or
                    #     maxNode[1] < miniBbox[1] or miniNode[1] > maxBbox[1] or
                    #     maxNode[2] < miniBbox[2] or miniNode[2] > maxBbox[2] ):
                        # cuda.atomic.add(results, node_number, 1)
                    results[thread,node_number]= not    (maxNode[0] < miniBbox[0] or miniNode[0] > maxBbox[0] or
                                                        maxNode[1] < miniBbox[1] or miniNode[1] > maxBbox[1] or
                                                        maxNode[2] < miniBbox[2] or miniNode[2] > maxBbox[2] )
 



    #find bounding Nodes for one layer with cuda                
    def findBoundingNodesOnce_cuda(self,target_bounding_boxes,intersected_nodes = None):
        if intersected_nodes is None:
            centers, sizes = OctreeOperator.maxtrixOperator(self.all_leaf_nodes(self.root.children[0]))
        else:
            centers, sizes = OctreeOperator.maxtrixOperator(intersected_nodes)
            
        # print(centers.shape)
        # print(sizes.shape)
        # 将数据传递到 GPU
        # 将数据传递到 GP
        
        centers_gpu = cuda.to_device(np.array(centers))
        sizes_gpu = cuda.to_device(np.array(sizes))
        target_bounding_box_gpu = cuda.to_device(np.ascontiguousarray(target_bounding_boxes))
        result_gpu = cuda.device_array((target_bounding_box_gpu.shape[0],centers_gpu.shape[0]),dtype=np.int8)
        output_test_gpu = cuda.device_array(target_bounding_box_gpu.shape[0],dtype=np.int8)
        # 设置线程和块大小
        threads_per_block = 256
        blocks_per_grid = (target_bounding_box_gpu.shape[0] + threads_per_block - 1) // threads_per_block
        
        # 调用 CUDA 核函数
        OctreeOperator.boundingBoxIntersect_cuda[blocks_per_grid, threads_per_block](centers_gpu, sizes_gpu, target_bounding_box_gpu, result_gpu, output_test_gpu)
        
        # 从 GPU 获取结果
        # cuda.synchronize()
        print("at 1")
        begin = time.time()
        result_2d= result_gpu.copy_to_host()
        end = time.time()
        duration = end - begin
        print("at 2")
        print("amount:",centers_gpu.shape[0],"get result time",duration, "storage:",result_2d.nbytes/(1024*1024*1024))
        result = OctreeOperator.reduce_size(result_2d)
        # for i in output_test_gpu.copy_to_host():
        #     print(i)
        return result
    
    # find all BoundingNodes of target depth with cuda
    def findBoundingNodesAll_cuda(self, target_depth, target_bounding_boxes,intersected_nodes=None):
        if intersected_nodes is None:
            intersected_tree_nodes = self.root.children
        else:
            intersected_tree_nodes = intersected_nodes
        max_depth = 2
        results = np.zeros(len(intersected_tree_nodes), dtype=np.int8)
        while max_depth <= target_depth:
            print("go in")
            results = self.findBoundingNodesOnce_cuda(target_bounding_boxes=target_bounding_boxes,intersected_nodes=intersected_tree_nodes)
            print("go out")
            for i in range(results.size):
                if results[i]:
                    OctreeOperator.extend(intersected_tree_nodes[i],intersected_tree_nodes[i].depth+1)
            intersected_tree_nodes = self.intersected_node_update(results,intersected_tree_nodes)
            
            max_depth = intersected_tree_nodes[0].depth
            # self.all_leaf_nodes()
            
            # print(max_depth)
        print("end")
        return intersected_tree_nodes

    
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
    @staticmethod
    def reduce_size(arr):
    # 沿着第一维度（轴0）应用any函数
        result = np.any(arr, axis=0)
        return result
    def intersected_node_update(self,results,intersected_nodes_old):
        intersected_tree_nodes_new = []
        for i in range(results.size):
            if results[i]:
                intersected_tree_nodes_new.extend(intersected_nodes_old[i].children)
        return intersected_tree_nodes_new
if __name__ == "__main__":
    # data_path = "B:\Master arbeit\DONUT2.stl"
    data_path = "B:\Master arbeit\Loopy Looper Donuts.stl"
    voxl = Voxlization(data_path)
    triangles = voxl.read_stl()
    bounding_boxes = np.array(voxl.boundingBoxes_gpu(triangles))
    maxBoundingBox = voxl.maxBoundingBox()
    octreeTest = OctreeOperator(maxBoundingBox)
    targetboundingbox =bounding_boxes[0].T
    targetboundingboxes =np.transpose(bounding_boxes, (0, 2, 1))

    # OctreeOperator.boundingBoxIntersect_cuda[blocks_per_grid, threads_per_block](centers_gpu, sizes_gpu, target_bounding_box_gpu, result_gpu)

    # for boundingbox in targetboundingboxes:
    #         center = (boundingbox[0] + boundingbox[1]) / 2
    #         size = boundingbox[1] - boundingbox[0]
    #         if np.any(size <= 0):
    #             print(boundingbox)
    #             raise ValueError("smaller than 0")
    

#     profiler = cProfile.Profile()

# # 开始性能分析
#     profiler.enable()

    # 运行你的函数
    # intersected_nodes = octreeTest.findBoundingNode_recursion(target_depth=10, target_bounding_box=targetboundingbox)
    
    intersected_nodes = octreeTest.findBoundingNodesAll_cuda(target_depth=8,target_bounding_boxes=targetboundingboxes,intersected_nodes=[octreeTest.root.children[0]])
    
    
    octreeTest.all_leaf_nodes()
    # print("GPU Duration")
    resolution = octreeTest.root.size*(0.5**10)
    print("resolution",resolution)
    print("how many leafnodes",len(octreeTest.leafnodes))
    print("how many intersected nodes",len(intersected_nodes))
    # octreeTest.visualize()
    

    # octreeTest.all_leaf_nodes()   

    # 结束性能分析
    # profiler.disable()

    # # 生成性能报告
    # profiler.print_stats(sort='cumulative')  # 按照累积时间排序