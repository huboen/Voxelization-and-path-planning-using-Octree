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
    def insert(self, point):
        pass
    def __buildTree__(self,node, depth):
        if depth == 0:
            return
        node.childNode()
        for child_node in node.children:
            self.__buildTree__(child_node, depth - 1)
        
    def __extend__(self,node,depth=1):
        if depth <= 0:
            return
        if not node.children:
        # If the node doesn't have children, create them
            node.childNode()
        else:
        # If the node already has children, recursively extend each child with reduced depth
            for child_node in node.children:
                self.__extend__(child_node,depth-1)
    def leafNodeCount(self):
        pass
    def search(self,point):
        pass
    def delPoint(self,point):
        pass
    def traversal(self):
        pass
    def visualize(self):
        pass
    def depth(self,node):
        pass
    def update(self,node):
        pass
    def center(self,boundingbox):
        center = [0.5*(boundingbox[0][0]+boundingbox[1][0]),0.5*(boundingbox[0][1]+boundingbox[1][1]),0.5*(boundingbox[0][2]+boundingbox[1][2])]
        return center    

        
        
if __name__ == "__main__":
    bounding_box = np.array([[0, 0, 0], [1, 1, 1]])
    octree = Octree(bounding_box)
    print("hello")
 