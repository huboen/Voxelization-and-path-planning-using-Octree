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
import math
from Cuda_operator import *

class BoundingboxTool:
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
            boundingBoxGroup.append(BoundingboxTool.boundingBox(triangle))
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

    def minBoundingBox(self):
        triangles = self.read_stl()
        bounding_boxes = np.array(voxl.boundingBoxes_gpu(triangles))
        size = np.min(np.max(bounding_boxes[:,:,1]-bounding_boxes[:,:,0],axis=1))
        return size
        
        
 

    

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
                miniBbox = cuda.local.array(3, dtype=numba.float32)
                maxBbox = cuda.local.array(3, dtype=numba.float32)
                miniBbox, maxBbox = bbox_slice[0], bbox_slice[1]
                for node_number in range(len(centers)):
                    miniNode = cuda.local.array(3, dtype=numba.float32)
                    maxNode = cuda.local.array(3, dtype=numba.float32)
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
        # print("at 1")
        begin = time.time()
        result_2d= result_gpu.copy_to_host()
        end = time.time()
        duration = end - begin
        # print("at 2")
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
        max_depth = intersected_tree_nodes[0].depth
        results = np.ones(len(intersected_tree_nodes), dtype=np.int8)
        while max_depth < target_depth:
            for i in range(results.size):
                if results[i]:
                    OctreeOperator.extend(intersected_tree_nodes[i],intersected_tree_nodes[i].depth+1)
            intersected_tree_nodes = self.intersected_node_update(results,intersected_tree_nodes)
            # print("go in")
            results = self.findBoundingNodesOnce_cuda(target_bounding_boxes=target_bounding_boxes,intersected_nodes=intersected_tree_nodes)
            # print("go out")
            
            
            max_depth = intersected_tree_nodes[0].depth
            # self.all_leaf_nodes()
        # intersected_tree_nodes = np.where(results==True,intersected_tree_nodes,0)
        intersected_tree_nodes = np.delete(intersected_tree_nodes, np.where(results==False) ) 
            # print(max_depth)
        # print("end")
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
                if len(intersected_nodes_old[i].children) :
                    intersected_tree_nodes_new.extend(intersected_nodes_old[i].children)
                else:
                    intersected_tree_nodes_new.extend(intersected_nodes_old)
        return intersected_tree_nodes_new
    @staticmethod
    def transferNode2box(nodes):
        boundingBoxes = []
        for node in nodes:
            min_bound = np.array(node.center)- node.size
            max_bound = np.array(node.center)+ node.size
            boundingBoxes.append([min_bound,max_bound])
        return boundingBoxes
    
class separatingAxis:
    def __init__(self) -> None:
        pass

    @staticmethod
    def overlap_on_axis(triangle_proj, box_proj):
    # 判断投影是否重叠
        return max(triangle_proj[0], box_proj[0]) <= min(triangle_proj[1], box_proj[1])
    @staticmethod
    def project_calculation(points, axis):
        # 计算所有点在轴上的投影，并返回最小和最大值
        values = [np.dot(point, axis) for point in points]
        return min(values), max(values)
    @staticmethod
    def collision_3d_triangle_box(triangle, box):
        # 计算三角形的法向量
        tri_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        tri_normal /= np.linalg.norm(tri_normal)
        triangle_edges = [triangle[1] - triangle[0],triangle[2] - triangle[0],triangle[2] - triangle[1]]
        box_edges = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,0,1]]
        # 计算 box 中心到平面的距离
        box_center = (box[0] + box[1]) / 2
        box_dimension = (box[1]-box[0])/2
        box_points = []
        for box_edge in box_edges:
            box_dimension_copy = box_dimension.copy()
            box_dimension_copy[box_edge == 0] = 0
            box_points.append(box_center+box_dimension_copy)
        distance_to_plane = np.abs(np.dot(box_center - triangle[0], tri_normal))
        # 计算 box 的投影半径
        box_radius = np.abs(np.dot(box[1] - box[0], tri_normal))
        if distance_to_plane < box_radius:
        # 计算三角形一边与包围盒一边的叉乘向量
            for triangle_edge in triangle_edges:
                for box_edge in box_edges:
                    edge_cross = np.cross(triangle_edge, box_edge)
                            # 计算三角形在叉乘向量上的投影
                    tri_proj = separatingAxis.project_calculation([triangle[0], triangle[1], triangle[2]], edge_cross)

                    # 计算包围盒在叉乘向量上的投影
                    box_proj = separatingAxis.project_calculation([box_points[0], box_points[1], box_points[2], box_points[3]], edge_cross)
                    if not separatingAxis.overlap_on_axis(tri_proj, box_proj):
                        # 三角形和包围盒在叉乘向量上的投影不重叠或距离超过半径，不相交
                        return False
        # 三角形和包围盒相交
        return True
    

    @staticmethod                      
    @cuda.jit
    def collision_3d_triangle_box_cuda(triangles, nodes,intersected_nodes):
        tri_thread,node_thread= cuda.grid(2)
        if tri_thread < triangles.shape[0] and node_thread < nodes.shape[0]:
            triangle = triangles[tri_thread]
            node = nodes[node_thread]
            triangle_edges = triangle.edges
            tri_normal = triangle.normal
            node_edges = [[1,0,0],[0,1,0],[0,0,1]]
            node_center = node.center
            node_dimension = [node.size/2, node.size/2, node.size/2] 
            node_vertex = node.vertex
            vector = cuda.local.array(3,dtype=numba.float32)
            sub(node_center,triangle.points[0],vector)
            distance2plane = cuda.local.array(1,dtype=numba.float32)
            distance_to_plane(vector,tri_normal,distance2plane)
            node_radius = cuda.local.array(1,dtype=numba.float32)
            dot(node_dimension,tri_normal,node_radius)
            node_radius[0] = math.fabs(node_radius[0])
            if distance_to_plane[0] < node_radius[0]:
                normal_vectors = cuda.local.array(9,dtype=numba.float32)
                tri_proj = cuda.local.array([3,9,1],dtype=numba.float32)
                box_proj = cuda.local.array([8,9,1],dtype=numba.float32)
                normal_vector(triangle_edges, node_edges,normal_vectors)
                        # 计算三角形在叉乘向量上的投影
                project_calculation_cuda(triangle.points, normal_vectors,tri_proj)

                # 计算包围盒在叉乘向量上的投影
                project_calculation_cuda(node.vertex, normal_vectors,box_proj)
            
                


if __name__ == "__main__":
    def transferNode2box(nodes):
        boundingBoxes = []
        for node in nodes:
            min_bound = np.array(node.center)- node.size
            max_bound = np.array(node.center)+ node.size
            boundingBoxes.append([min_bound,max_bound])
        return boundingBoxes
    data_path = "B:\Master arbeit\DONUT2.stl"
    # data_path = "B:\Master arbeit\Loopy Looper Donuts.stl"
    voxl = BoundingboxTool(data_path)
    triangles = voxl.read_stl()
    bounding_boxes = np.array(voxl.boundingBoxes_gpu(triangles))
    maxBoundingBox = voxl.maxBoundingBox()
    minBoundingBox = voxl.minBoundingBox()
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
    target_depth= 2
    start = time.time()
    intersected_nodes = octreeTest.findBoundingNodesAll_cuda(target_depth=2,target_bounding_boxes=targetboundingboxes,intersected_nodes=[octreeTest.root.children[1]])
    depth = intersected_nodes[0].depth
    while depth<target_depth:
        intersected_nodes=np.array_split(intersected_nodes, np.ceil(len(intersected_nodes) /5000))
        a = []
        for group in intersected_nodes:
            a.extend(octreeTest.findBoundingNodesAll_cuda(target_depth=depth+1,target_bounding_boxes=targetboundingboxes,intersected_nodes=group))
        depth = intersected_nodes[0].depth
    end = time.time()
    duration = end - start
    octreeTest.all_leaf_nodes()
    resolution = octreeTest.root.size*(0.5**(target_depth))
    print("Computation time", duration)
    print("resolution",resolution)
    print("how many leafnodes",len(octreeTest.leafnodes))
    print("how many intersected nodes",len(intersected_nodes))
    node_boxes = transferNode2box(intersected_nodes)
    octreeTest.visualize(stl_path=data_path,boundingboxes=node_boxes,octree=False)
    

    # octreeTest.all_leaf_nodes()   

    # 结束性能分析
    # profiler.disable()

    # # 生成性能报告
    # profiler.print_stats(sort='cumulative')  # 按照累积时间排序