import numpy as np
import open3d as o3d
class Octreenode:
    def __init__(self,center,size,parent=None) -> None:
        self.center = center
        self.size = size
        self.label = 0 
        self.children = []
        self.parent = parent
        self.depth = self.get_depth()

    def childNode(self):
        child_size = self.size / 2
        half_size = child_size / 2
        child_centers = [
            [self.center[0] - half_size, self.center[1] - half_size, self.center[2] - half_size],
            [self.center[0] - half_size, self.center[1] - half_size, self.center[2] + half_size],
            [self.center[0] - half_size, self.center[1] + half_size, self.center[2] - half_size],
            [self.center[0] - half_size, self.center[1] + half_size, self.center[2] + half_size],
            [self.center[0] + half_size, self.center[1] - half_size, self.center[2] - half_size],
            [self.center[0] + half_size, self.center[1] - half_size, self.center[2] + half_size],
            [self.center[0] + half_size, self.center[1] + half_size, self.center[2] - half_size],
            [self.center[0] + half_size, self.center[1] + half_size, self.center[2] + half_size],
        ]
        self.children = [Octreenode(i, size=child_size,parent=self) for i in child_centers]

    def get_depth(self):
        # calculate the depth recursively
        if self.parent is None:
            return 0  # if root, then 0
        else:
            return 1 + self.parent.depth

    def get_node_info(self):
        return (self.center, self.size, self.get_depth())  


class Octree:
    def __init__(self,boundingbox) -> None:
        center = self.center(boundingbox)
        initial_size = np.max(boundingbox[1]-boundingbox[0])
        self.root = Octreenode(center,size=initial_size)
        self.leafnodes = []
        self.__buildTree__(self.root,depth=1)
    #initial the octree
    def __buildTree__(self,node, depth):
        if depth == 0:
            return
        node.childNode()
        for child_node in node.children:
            self.__buildTree__(child_node, depth - 1)
    # extend one node to given depth
    @staticmethod   
    def extend(node, target_depth):
        current_depth = node.depth
        
        if current_depth == target_depth:
            # print("Already at the target depth")
            return
        elif current_depth > target_depth:
            # print("Cannot extend to a shallower depth")
            return

        if not node.children:
            node.childNode()

        for child_node in node.children:
            Octree.extend(child_node, target_depth)
    # insert points in to the tree
    def insert(self, point):
        pass
    def leafNodeCount(self):
        pass

    def find_leaf_node(self, point, node=None):
        if node == None:
            node = self.root
            if not self.point_inside_box(point,node.center,node.size):
                raise ValueError("Point not in the octree")
    # if no children for node, it is leaf
        if not node.children:
            return node
        for child_node in node.children:
            if self.point_inside_box(point, child_node.center, child_node.size):
                return self.find_leaf_node(point, child_node)
    @staticmethod
    def point_inside_box(point, center, size):
        return all(center[i] - size / 2 <= point[i] <= center[i] + size / 2 for i in range(3))


    def delPoint(self,point):
        pass
    def __update_leaf_nodes__(self,node=None):
        if node == None:
            node = self.root
        if not node.children :
            self.leafnodes.append(node)
            return
        for child in node.children:
            self.__update_leaf_nodes__(child)
    #show all leaf nodes
    def all_leaf_nodes(self,target_node = None):
        self.leafnodes = []
        self.__update_leaf_nodes__(node= target_node)
        return self.leafnodes
        
        
    #visualize with open3d
    def load_stl_with_border(self, stl_path):
        mesh = o3d.io.read_triangle_mesh(stl_path)

        # 创建 STL 文件的边框
        lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        lines.paint_uniform_color([1, 0, 0])  # 边框的颜色，可以根据需要进行调整

        return mesh, lines

    # 新增一个方法来将 STL 几何体和边框添加到可视化中
    def add_stl_with_border_to_visualization(self, stl_path, line_set_list):
        stl_mesh, border_lines = self.load_stl_with_border(stl_path)

        # 添加 STL 文件的边框
        line_set_list.append(border_lines)

        # 添加 STL 文件的实体几何体
        line_set_list.append(stl_mesh)
    
    @staticmethod
    def create_cuboid_from_corners(boundingboxes,line_set_list):
        for boundingbox in boundingboxes:
            center = (boundingbox[0] + boundingbox[1]) / 2
            size = np.abs(boundingbox[1] - boundingbox[0])
            if np.any(size<= 0):
                continue
            cuboid = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2])
            cuboid.translate(boundingbox[0])
            cuboid.paint_uniform_color([1, 0, 0])
            line_set_list.append(cuboid)

    # 重写 visualize 方法以显示 STL 文件和边框
    def visualize(self, stl_path=None,boundingboxes=None):
        line_set_list = []
        self.__create_lines__(self.root, line_set_list)

        # 如果提供了 STL 文件路径，将其加载并添加到可视化中
        if stl_path:
            self.add_stl_with_border_to_visualization(stl_path, line_set_list)
        if boundingboxes is not None:
            self.create_cuboid_from_corners(boundingboxes,line_set_list)

        o3d.visualization.draw_geometries(line_set_list, window_name="Octree Visualization", width=800, height=600)

    def __create_lines__(self, node, line_set_list):
        if not node.children:
            # Create lines representing the bounding box edges
            edges = [
                [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7],
                [4, 5], [4, 6], [5, 7], [6, 7]
            ]
            points = [
                [node.center[0] - node.size / 2, node.center[1] - node.size / 2, node.center[2] - node.size / 2],
                [node.center[0] - node.size / 2, node.center[1] - node.size / 2, node.center[2] + node.size / 2],
                [node.center[0] - node.size / 2, node.center[1] + node.size / 2, node.center[2] - node.size / 2],
                [node.center[0] - node.size / 2, node.center[1] + node.size / 2, node.center[2] + node.size / 2],
                [node.center[0] + node.size / 2, node.center[1] - node.size / 2, node.center[2] - node.size / 2],
                [node.center[0] + node.size / 2, node.center[1] - node.size / 2, node.center[2] + node.size / 2],
                [node.center[0] + node.size / 2, node.center[1] + node.size / 2, node.center[2] - node.size / 2],
                [node.center[0] + node.size / 2, node.center[1] + node.size / 2, node.center[2] + node.size / 2]
            ]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(edges)
            line_set_list.append(line_set)
            # Create a cuboid representing the bounding box
            #array([0.00794744, 0.0864054 , 0.06098304], dtype=float32)
            #array([-0.82648194, -1.2520468 ,  0.01052197], dtype=float32)
            #width=0.007947445, height=0.06098304, depth=0.0864054
            cuboid = o3d.geometry.TriangleMesh.create_box(width=0.007947445, height=0.0864054, depth=0.06098304)
            cuboid.translate([-0.83045566, -1.2952495 , -0.01996955])
            # cuboid.translate([-1.5419416427612305, -1.2957134246826172, 0.01389758288860321])
            cuboid.paint_uniform_color([1, 0, 0])
            line_set_list.append(cuboid)
        else:
            for child_node in node.children:
                self.__create_lines__(child_node, line_set_list)
        # create axis
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # axis.translate(node.center)
        # line_set_list.append(axis)
    def update(self,node):
        pass
    #calculate the maximum depth of the octree
    def max_depth(self):
        self.all_leaf_nodes
        depth = np.array(list(node.depth for node in self.leafnodes))
        maxDepth = np.max(depth)
        return maxDepth

    def center(self,boundingbox):
        center = [0.5*(boundingbox[0][0]+boundingbox[1][0]),0.5*(boundingbox[0][1]+boundingbox[1][1]),0.5*(boundingbox[0][2]+boundingbox[1][2])]
        return center    

        
        
if __name__ == "__main__":
    bounding_box = np.array([[0, 0, 0], [1, 1, 1]])

    # 创建八叉树
    octree = Octree(bounding_box)
    test_point = [0.45, 0.45, 0.45]
    test_node = octree.find_leaf_node(test_point)
    root_node = octree.root
    # print(test_node.depth())
    for i in root_node.children:
        target_depth = np.random.randint(2,5)
        # print(target_depth)
        octree.extend(i,target_depth = target_depth)
    test_node = octree.find_leaf_node(test_point)
    print(octree.max_depth())
    data_path = "B:\Master arbeit\DONUT2.stl"
    octree.visualize(data_path)

    # # 插入一些点
    # points_to_insert = [
    #     [0.2, 0.2, 0.2],
    #     [0.8, 0.8, 0.8],
    #     [0.5, 0.5, 0.5],
    # ]

    # for point in points_to_insert:
    #     octree.insert(point)

    # # 查找叶子节点
    # test_point = [0.7, 0.7, 0.7]
    # leaf_node = octree.find_leaf_node(test_point)
    # print("Leaf Node Center:", leaf_node.center)
    # print("Leaf Node Size:", leaf_node.size)

    # # 尝试查找一个不在八叉树中的点
    # unknown_point = [1.2, 1.2, 1.2]
    # try:
    #     octree.find_leaf_node(unknown_point)
    # except ValueError as e:
    #     print("Error:", e)