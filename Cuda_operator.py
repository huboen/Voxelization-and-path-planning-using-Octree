from numba import cuda
import numpy as np
import numba
import math
@cuda.jit(device=True)
def dot(a, b, result):
    for i in range(a.shape[0]):
        result[0]+=a[i]*b[i]
@cuda.jit(devide=True)
def sum(a,b,result):
    for i in range(a.shape[0]):
        result[i] = a[i] + b[i]
@cuda.jit(device=True)
def sub(a,b,result):
    for i in range(a.shape[0]):
        result[i] = a[i] - b[i]
@cuda.jit(device=True)
def cross(a,b,result):
    for i in range(a.shape[0]):
        if i == 0:
            result[i] = a[1] * b[2] - a[2] * b[1]
        if i == 1:
            result[i] = -a[0] * b[2] + a[2] * b[0]
        if i == 2:
            result[i] = a[0] * b[1] - a[1] * b[0]

@cuda.jit(device=True)
def triangle_edge(points,edges):
    i = cuda.threadIdx.x
    for i in range(edges.shape[0]):
        if i == 0:
            sub(points[1],points[2],edges[i])
        if i == 1:
            sub(points[0],points[2],edges[i])
        if i == 2:
            sub(points[0],points[1],edges[i])

@cuda.jit      
def distance_to_plane(vec,normal_vec,result):
    dot(vec,normal_vec,result)
    result[0] = math.fabs(result[0])


@cuda.jit
def normal_vector(edges1, edges2, result):
    i, j = cuda.grid(2)
    if i <edges1.shape[0] and j <edges2.shape[0]:
            result[i,j, 0] = edges1[i, 1] * edges2[j, 2] - edges1[i, 2] * edges2[j, 1]
            result[i,j, 1] = -edges1[i, 0] * edges2[j, 2] + edges1[i, 2] * edges2[j, 0]
            result[i,j, 2] = edges1[i, 0] * edges2[j, 1] - edges1[i, 1] * edges2[j, 0]
            

@cuda.jit
def project_calculation_cuda(points,axises,result):
    i,j= cuda.grid(2)
    if i < points.shape[0] and j < axises.shape[0]:
        dot(points[i],axises[j],result[i,j])



@cuda.jit
def collision_3d_triangle_box_cuda(triangles_vertex,triangles_normals,nodes_center,nodes_vertex,node_size, results):
    
    tri_thread,node_thread= cuda.grid(2)
    if tri_thread < triangles_normals[0].shape[0] and node_thread < nodes_center[0].shape[0]:
        d_triangle_edges = cuda.local.array((3,3),dtype=np.float32)
        triangle_edge(triangles_vertex[tri_thread],results[tri_thread])
        tri_normal = triangles_normals[tri_thread]
        node_edges = [[1,0,0],[0,1,0],[0,0,1]]
        d_node_center = nodes_center[node_thread]
        d_node_vertex = nodes_vertex[node_thread]
        d_tri_vertex = triangles_vertex[tri_thread]
        d_node_dimension = [node_size[node_thread]/2,node_size[node_thread]/2,node_size[node_thread]/2]
        vector = cuda.local.array(3,dtype=numba.float32)
        sub(d_node_center,d_tri_vertex[0],vector)
        distance2plane = cuda.local.array(1,dtype=numba.float32)
        distance_to_plane(vector,tri_normal,distance2plane)
        # node_radius = cuda.local.array(1,dtype=numba.float32)
        # tri_normal_abs= [math.fabs(tri_normal[tri_thread,0]),math.fabs(tri_normal[tri_thread,1]),math.fabs(tri_normal[tri_thread,2])]
        # dot(d_node_dimension,tri_normal_abs,node_radius)
        # if distance_to_plane[0] < node_radius[0]:
        #     normal_vectors = cuda.local.array(9,dtype=numba.float32)
        #     tri_proj = cuda.local.array([3,9,1],dtype=numba.float32)
        #     box_proj = cuda.local.array([8,9,1],dtype=numba.float32)
        #     normal_vector(d_triangle_edges, node_edges,normal_vectors)
        #             # 计算三角形在叉乘向量上的投影
        #     project_calculation_cuda(d_tri_vertex, normal_vectors,tri_proj)

        #     # 计算包围盒在叉乘向量上的投影
        #     project_calculation_cuda(d_node_vertex, normal_vectors,box_proj)
            

