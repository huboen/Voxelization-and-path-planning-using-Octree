from Voxelization import Voxelization_exe,OctreeOperator
from Voxelization import OctreeOperator
import numpy as np
data_path = "B:\Master arbeit\DONUT2.stl"
# data_path = "B:\Master arbeit\Loopy Looper Donuts.stl"
DEPTH = 6
test = Voxelization_exe(data_path)
test.run(targetDepth=DEPTH,excel = True, vis = False)
