from pyoctree import pyoctree as ot
import numpy as np
bbox = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]

# 创建八叉树对象
octree = ot.PyOctree(bbox, depth=5)
