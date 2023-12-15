from Voxelization import Voxelization_exe
from Voxelization import OctreeOperator

data_path = "B:\Master arbeit\DONUT2.stl"
# data_path = "B:\Master arbeit\Loopy Looper Donuts.stl"
DEPTH = 8
test = Voxelization_exe(data_path)
intersected_nodes,inner_nodes,octree =test.run(targetDepth=DEPTH,excel = True, vis = False)
octree.update(intersected_nodes,inner_nodes)
slice = OctreeOperator.node_division(octree)


        

