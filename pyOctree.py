import numpy as np
import open3d as o3d
class Octreenode:
    def __init__(self,center,size,parent=None) -> None:
        self.center = center
        self.size = size
        self.label = 0 
        self.children = []
        self.parent = parent

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

    def depth(self):
        # calculate the depth recursively
        if self.parent is None:
            return 0  # if root, then 0
        else:
            return 1 + self.parent.depth()  


class Octree:
    def __init__(self,boundingbox) -> None:
        center = self.center(boundingbox)
        self.root = Octreenode(center,size=0.1 )
        self.__buildTree__(self.root,depth=1)
    #initial the octree
    def __buildTree__(self,node, depth):
        if depth == 0:
            return
        node.childNode()
        for child_node in node.children:
            self.__buildTree__(child_node, depth - 1)
    # extend one node to given depth   
    def extend(self, node, target_depth):
        current_depth = node.depth()
        
        if current_depth == target_depth:
            print("Already at the target depth")
            return
        elif current_depth > target_depth:
            print("Cannot extend to a shallower depth")
            return

        if not node.children:
            node.childNode()

        for child_node in node.children:
            self.extend(child_node, target_depth)
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
    def traversal(self):
        pass
    #visualize with open3d
    def visualize(self):
        line_set_list = []
        self.__create_lines__(self.root, line_set_list)
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
        else:
            for child_node in node.children:
                self.__create_lines__(child_node, line_set_list)
        # create axis
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # axis.translate(node.center)
        # line_set_list.append(axis)
    def update(self,node):
        pass
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
        print(target_depth)
        octree.extend(i,target_depth = target_depth)
    test_node = octree.find_leaf_node(test_point)
    octree.visualize()

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