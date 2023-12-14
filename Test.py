from Cuda_operator import * 
import time
from Voxelization import separatingAxis
class node:
     def __init__(self,center,size) -> None:
          self.center = center
          self.size = size

def node_vertex(node_centers,size):
    vertexes = np.zeros((node_centers.shape[0],8,3),dtype=np.float32)
    for i in range(node_centers.shape[0]):
        vertex = np.array([[node_centers[i,0]- size[i]/2,node_centers[i,1] - size[i]/2,node_centers[i,2] - size[i]/2],
                        [node_centers[i,0]- size[i]/2,node_centers[i,1] - size[i]/2,node_centers[i,2] + size[i]/2],
                        [node_centers[i,0]- size[i]/2,node_centers[i,1] + size[i]/2,node_centers[i,2] - size[i]/2],
                        [node_centers[i,0]- size[i]/2,node_centers[i,1] + size[i]/2,node_centers[i,2] + size[i]/2],
                        [node_centers[i,0]+ size[i]/2,node_centers[i,1] - size[i]/2,node_centers[i,2] - size[i]/2],
                        [node_centers[i,0]+ size[i]/2,node_centers[i,1] - size[i]/2,node_centers[i,2] + size[i]/2],
                        [node_centers[i,0]+ size[i]/2,node_centers[i,1] + size[i]/2,node_centers[i,2] - size[i]/2],
                        [node_centers[i,0]+ size[i]/2,node_centers[i,1] + size[i]/2,node_centers[i,2] + size[i]/2]],dtype=np.float32)
        vertexes[i]=vertex
    return vertexes 
@cuda.jit
def test(edges1,edges2,results):
    normal_vector(edges1,edges2,results)

@cuda.jit
def test2(d_tri_vertex,d_node_vertex,d_node_edges,axis,result,tri_proj,node_proj):
        d_triangle_edges = cuda.local.array((3,3),dtype=np.float32)
        triangle_edge(d_tri_vertex,d_triangle_edges)
        # overlap_on_axis(d_tri_vertex,d_node_vertex,tri_normal,tri_proj,node_proj,test)
        project_calculation_cuda(d_tri_vertex,tri_normal,tri_proj)
        project_calculation_cuda(d_node_vertex,axis,node_proj)
        # result[0] = 8
        result[0]=overlap_on_axis(d_tri_vertex,d_node_vertex,tri_normal,tri_proj,node_proj)
        if not overlap_on_axis(d_tri_vertex,d_node_vertex,tri_normal,tri_proj,node_proj):
            return
        for axis in d_triangle_edges:
            result[0] = 8
            if not overlap_on_axis(d_tri_vertex,d_node_vertex,axis,tri_proj,node_proj):
                return
        # for axis in d_node_edges:
        #     if not overlap_on_axis(d_tri_vertex,d_node_vertex,axis,tri_proj,node_proj):
        #         return
        # for d_tri_edge in d_triangle_edges:
        #     for d_node_edge in d_node_edges:
        #         axis = cuda.local.array(3,dtype=np.float32)
        #         cross(d_tri_edge,d_node_edge,axis)
        #         if not overlap_on_axis(d_tri_vertex,d_node_vertex,axis,tri_proj,node_proj):
        #             return
        result[0] = 8
        result[0]=overlap_on_axis(d_tri_vertex,d_node_vertex,tri_normal,tri_proj,node_proj)


if __name__ == "__main__":
    node1 = node([-3.4169416427612305, -3.170713424682617, -1.8611024171113968],1.25)
    tri_vertex=np.array([[-0.8225082 , -1.2952495 , -0.0189103 ],
        [-0.827361  , -1.2088441 , -0.01996955],
        [-0.83045566, -1.2090623 ,  0.04101349]], dtype=np.float32)
    tri_normal = np.array([ 5.2690329e-03,  2.9921482e-04,  2.6845571e-04], dtype=np.float32)
    node_vertexes =     np.array(   [[-4.0419416427612305, -3.795713424682617, -2.486102417111397],
                                      [-4.0419416427612305, -3.795713424682617, -1.2361024171113968],
                                        [-4.0419416427612305, -2.545713424682617, -2.486102417111397],
                                          [-4.0419416427612305, -2.545713424682617, -1.2361024171113968],
                                            [-2.7919416427612305, -3.795713424682617, -2.486102417111397],
                                              [-2.7919416427612305, -3.795713424682617, -1.2361024171113968],
                                                [-2.7919416427612305, -2.545713424682617, -2.486102417111397],
                                                  [-2.7919416427612305, -2.545713424682617, -1.2361024171113968]], dtype=np.float32)
    node_edges = np.array([[1,0,0],[0,0,1],[0,1,0]])
    d_tri_vertex = cuda.to_device(tri_vertex)
    d_tri_normal = cuda.to_device(tri_normal)
    d_node_vertexes = cuda.to_device(node_vertexes)
    d_node_edges = cuda.to_device(node_edges)
    d_result = cuda.device_array(1,dtype=np.int8)
    tri_proj = cuda.device_array(3,dtype=np.float32)
    node_proj = cuda.device_array(8,dtype=np.float32)
    test2[1,1](d_tri_vertex,d_node_vertexes,d_node_edges,tri_normal,d_result,tri_proj,node_proj)
    result = d_result.copy_to_host()
    result2 = separatingAxis.collision_3d_triangle_box(tri_vertex,tri_normal,node1)
    tri_proj2 = []
    for vertex in tri_vertex:
         a = np.dot(vertex,tri_normal)
         
    tri_proj2 = [np.dot(vertex,tri_normal) for vertex in tri_vertex]
    node_vertexes = separatingAxis.node_vertexes_from_center_and_size(node1.center, node1.size)
    node_proj2 = [np.dot(node_vertex, tri_normal) for node_vertex in node_vertexes]
    if max(tri_proj2)<min(node_proj2) or min(tri_proj2) >max(node_proj2):
        print("not intersected")
    print(tri_proj.copy_to_host())
    print(node_proj.copy_to_host())
    print(tri_proj2)
    print(node_proj2)
    print(result,result2)
    
