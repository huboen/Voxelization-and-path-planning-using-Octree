from numba import cuda
import numpy as np
import numba
import math
@cuda.jit(device=True)
def dot(a, b):
    result = 0
    for i in range(a.shape[0]):
        result+=a[i]*b[i]
    return result
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
    for i in range(edges1.shape[0]):
        for j in range(edges2.shape[0]):
            result[i+j, 0] = edges1[i, 1] * edges2[j, 2] - edges1[i, 2] * edges2[j, 1]
            result[i+j, 1] = -edges1[i, 0] * edges2[j, 2] + edges1[i, 2] * edges2[j, 0]
            result[i+j, 2] = edges1[i, 0] * edges2[j, 1] - edges1[i, 1] * edges2[j, 0]
            

@cuda.jit
def project_calculation_cuda(points,axis,result):
    for i in range(points.shape[0]) :
        result[i] = dot(points[i],axis)

@cuda.jit(device=True)
def abs(vector,result):
    for i in range(len(vector)):
        result[i] = math.fabs(vector[i])

@cuda.jit(device=True)
def overlap_check(vectors1,vectors2,result):
    for i in range(vectors1.shape[1]):
        vector1_min = min(vectors1[:,i,0])
        vector1_max = max(vectors1[:,i,0])
        vector2_min = min(vectors2[:,i,0])
        vector2_max = max(vectors2[:,i,0])
        if vector1_max < vector2_min or vector1_min>vector2_max:
            pass
        else:
            result[0] = True

@cuda.jit(device = True)
def max_radius(normals,dimension,radius):
    radius[0] = normals[0]*dimension[0]/math.sqrt(normals[0]**2 +normals[1]**2 +normals[2]**2)
    radius[1] = normals[1]*dimension[1]/math.sqrt(normals[0]**2 +normals[1]**2 +normals[2]**2)
    radius[2] = normals[2]*dimension[2]/math.sqrt(normals[0]**2 +normals[1]**2 +normals[2]**2)



@cuda.jit(device = True)
def overlap_on_axis(triangle_vertex,nodes_vertex,axis,tri_proj,node_proj):
    project_calculation_cuda(triangle_vertex,axis,tri_proj)
    project_calculation_cuda(nodes_vertex,axis,node_proj)
    if max(tri_proj)<min(node_proj) or min(tri_proj)>max(node_proj):
        return False
    return True