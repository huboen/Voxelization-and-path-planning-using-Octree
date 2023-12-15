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
            [self.center[0] + half_size, self.center[1] - half_size, self.center[2] - half_size],
            [self.center[0] - half_size, self.center[1] + half_size, self.center[2] - half_size],
            [self.center[0] + half_size, self.center[1] + half_size, self.center[2] - half_size],
            [self.center[0] - half_size, self.center[1] - half_size, self.center[2] + half_size],
            [self.center[0] + half_size, self.center[1] - half_size, self.center[2] + half_size],
            [self.center[0] - half_size, self.center[1] + half_size, self.center[2] + half_size],
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
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Octree):
            return other.center == self.center


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
            size = (boundingbox[1] - boundingbox[0])/2
            if np.any(size<= 0):
                continue
            cuboid = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2])
            cuboid.translate(center,relative=False)
            cuboid.paint_uniform_color([0, 0, 1])
            line_set_list.append(cuboid)

    # 重写 visualize 方法以显示 STL 文件和边框
    def visualize(self, stl_path=None,boundingboxes=None,octree=True):
        line_set_list = []
        if octree:
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
            for child_node in node.children:
                self.__create_lines__(child_node, line_set_list)
    def max_depth(self):
        self.all_leaf_nodes
        depth = np.array(list(node.depth for node in self.leafnodes))
        maxDepth = np.max(depth)
        return maxDepth

    def center(self,boundingbox):
        center = 0.5*(boundingbox[0]+boundingbox[1])
        return center    
    
    def minimum_center_z(self,node = None, result = None):
        if result == None and node == None:
            node = self.root
            result = self.root.center[2]
        if not node.children:
            if result > node.center[2]:
                return node.center[2]
        for node in node.children[0:4]:
            if result >self.minimum_center_z(node,result):
                result = self.minimum_center_z(node,result)
        return result
    
    @staticmethod
    def update(intersected_nodes,inner_nodes):
        for node in intersected_nodes:
            node.label = 1
        for node in inner_nodes:
            node.label = 0.5



    

        
        
if __name__ == "__main__":
    pass