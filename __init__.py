from Voxelization import Voxelization_exe
from Voxelization import OctreeOperator
import openpyxl
import os

data_path = "B:\Master arbeit\DONUT2.stl"
# data_path = "B:\Master arbeit\Loopy Looper Donuts.stl"
DEPTH = 8
test = Voxelization_exe(data_path)
intersected_nodes,inner_nodes,octree =test.run(targetDepth=DEPTH,excel = True, vis = False)
octree.update(intersected_nodes,inner_nodes)
slice = OctreeOperator.node_division(octree)
current_dir = os.path.dirname(os.path.abspath(__file__))
for index in slice:
    name1="layer" + str(index) +".xlsx"
    file_path1 = os.path.join(current_dir,"layer_data", name1)
    OctreeOperator.write_to_excel(slice[index],name=file_path1)


        

