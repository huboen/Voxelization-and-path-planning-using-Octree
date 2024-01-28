from Voxelization import Voxelization_exe
from Voxelization import OctreeOperator
import openpyxl
import os


# data_path = "B:\Master arbeit\DONUT2.stl"
data_path = "B:\Master arbeit\Loopy Looper Donuts.stl"
DEPTH = 4
test = Voxelization_exe(data_path)
intersected_nodes,inner_nodes,octrees =test.run(targetDepth=DEPTH,excel = True, vis = True)
OctreeOperator.update(intersected_nodes,inner_nodes)
slice = OctreeOperator.node_division(octrees)
current_dir = os.path.dirname(os.path.abspath(__file__))

i = 0
for index in slice:
    name1="layer" + str(i) +".xlsx"
    file_path1 = os.path.join(current_dir,"layer_data", name1)
    OctreeOperator.write_to_excel(slice[index],name=file_path1)
    i += 1



        

