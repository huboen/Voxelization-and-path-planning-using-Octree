import numpy as np
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
        self.root = Octreenode(center,size=0.1)
        self.__buildTree__(self.root,depth=1)
    #initial the octree
    def __buildTree__(self,node, depth):
        if depth == 0:
            return
        node.childNode()
        for child_node in node.children:
            self.__buildTree__(child_node, depth - 1)
    # extend one node to given depth   
    def extend(self,node,depth):
        # if the node is already at the depth, return
        if node.depth() == depth:
            print("alreadly at the depth")
            return
        if not node.children:
        # If the node doesn't have children, create them
            node.childNode()
        else:
        # If the node already has children, recursively extend each child with reduced depth
            for child_node in node.children:
                self.extend(child_node,depth)
    # insert points in to the tree
    def insert(self, point):
        pass
    def leafNodeCount(self):
        pass

    def find_leaf_node(self, point, node=None):
        if node == None:
            node = self.root
            if not self.point_inside_box(point,node.center,node.size):
                print("point not in the octree")
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
    def visualize(self):
        pass
    def update(self,node):
        pass
    def center(self,boundingbox):
        center = [0.5*(boundingbox[0][0]+boundingbox[1][0]),0.5*(boundingbox[0][1]+boundingbox[1][1]),0.5*(boundingbox[0][2]+boundingbox[1][2])]
        return center    

        
        
if __name__ == "__main__":
    bounding_box = np.array([[0, 0, 0], [1, 1, 1]])

    # 创建八叉树
    octree = Octree(bounding_box)

    # 插入一些点
    points_to_insert = [
        [0.2, 0.2, 0.2],
        [0.8, 0.8, 0.8],
        [0.5, 0.5, 0.5],
    ]

    for point in points_to_insert:
        octree.insert(point)

    # 查找叶子节点
    test_point = [0.7, 0.7, 0.7]
    leaf_node = octree.find_leaf_node(test_point)
    print("Leaf Node Center:", leaf_node.center)
    print("Leaf Node Size:", leaf_node.size)

    # 尝试查找一个不在八叉树中的点
    unknown_point = [1.2, 1.2, 1.2]
    try:
        octree.find_leaf_node(unknown_point)
    except ValueError as e:
        print("Error:", e)