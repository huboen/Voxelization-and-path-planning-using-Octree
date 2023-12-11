from Cuda_operator import * 

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
def test2(points,edges):
    triangle_edge(points,edges)
if __name__ == "__main__":
#     array1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
#     array2 = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], dtype=np.float32)

# # 打印数组
#     threads_per_block = (16,16)
#     blocks_per_grid_x = (array1.shape[i,0]+ threads_per_block[i,0]- 1) // threads_per_block[0]
#     blocks_per_grid_y = (array2.shape[i,0]+ threads_per_block[i,1] - 1) // threads_per_block[i,1]
#     blocks_per_grid = (blocks_per_grid_x,blocks_per_grid_y)
#     d_array1 = cuda.to_device(array1)
#     d_array2 = cuda.to_device(array2)
#     d_results = cuda.device_array((array1.shape[0],array2.shape[0],1),dtype=np.float32)
#     test[blocks_per_grid,threads_per_block](d_array1,d_array2,d_results)
#     results = d_results.copy_to_host()
#     print("result")
#     print(results)
    triangles_vertex = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     [[10, 11, 12], [13, 14, 15], [16, 17, 18]]], dtype=np.float32)

    triangles_normals =np.array([[1,1,1],[2,1,1]],dtype=np.float32)
    nodes_center = np.array([[0,1,0],[3,4,5]],dtype=np.float32)
    node_size =  np.array([[1,1,1],[2,2,2]],dtype=np.float32)
    node_vertexes = node_vertex(nodes_center,node_size[0])
    node_edges = np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=np.int8)
    d_triangles_vertex = cuda.to_device(triangles_vertex)
    d_triangles_normals = cuda.to_device(triangles_normals)
    d_nodes_center = cuda.to_device(nodes_center)
    d_node_size = cuda.to_device(node_size)
    d_nodes_vertex = cuda.to_device(node_vertexes)
    d_node_edges = cuda.to_device(node_edges)
    threads_per_block = (16,16)
    blocks_per_grid_x = (triangles_normals.shape[0]+ threads_per_block[0]- 1) // threads_per_block[0]
    blocks_per_grid_y = (nodes_center.shape[0]+ threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x,blocks_per_grid_y)
    d_result =cuda.device_array((triangles_vertex.shape[0],3,3),dtype=np.float32)
    # test2[1,16](d_triangles_vertex[0],d_result[0])
    collision_3d_triangle_box_cuda[blocks_per_grid,threads_per_block](d_triangles_vertex,d_triangles_normals,d_nodes_center,d_nodes_vertex,d_node_size,d_node_edges,d_result)
    result = d_result.copy_to_host()
    print(result[0])