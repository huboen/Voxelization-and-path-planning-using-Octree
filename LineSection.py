import pandas as pd
import time
import open3d as o3d
import numpy as np
import copy
import math



class LineSection:
    def __init__(self,input) -> None:
        self.nodes = input["nodes"]
        self.range = input["range"]
        self.pos = input["y_position"]
        self.neighbors = []
        self.visited = False


class LineSectionRow:
    def __init__(self,pos) -> None:
        self.members = []
        self.pos = pos

    def insert(self,LineSection):
        if LineSection.pos == self.pos:
            self.members.append(LineSection)
        else:
            raise ValueError("The inserted pos doesn't match")


class LineSectionGroup:
    def __init__(self) -> None:
        self.members =[]

    def insert(self,LineSectionRow):
        self.members.append(LineSectionRow)
        self.__sort__()
    
    def __sort__(self):
        self.members = sorted(self.members,key=lambda x :x.pos)
    

class LineSectionOperator():
    def __init__(self,nodes_list) -> None:
        self.nodes = nodes_list

    def build_dic(self):
        nodes_dic = {}
        for node in self.nodes:
            if node.center[1] not in nodes_dic:
                nodes_dic[node.center[1]] = [node]
            else:
                nodes_dic[node.center[1]].append(node)
            
        return dict(sorted(nodes_dic.items()))

    @staticmethod
    def split_nodes_row(nodes):
        lineSecRow = LineSectionRow(nodes[0].center[1])
        sorted_nodes = sorted(nodes,key = lambda x:x.center[0])
        while sorted_nodes:
            line_sec = {"nodes":[],
                        "range":[float('-inf'),float('-inf')],
                        "y_position":nodes[0].center[1]}
            prev_node = None
            to_remove = []
            
            for node in sorted_nodes:
                if prev_node is None or math.isclose(node.center[0], prev_node.center[0]+prev_node.size, rel_tol=0.0001):
                    line_sec["nodes"].append(node)
                    line_sec["range"] = [line_sec["nodes"][0].center[0],line_sec["nodes"][-1].center[0]]
                    prev_node = copy.deepcopy(node)
                    to_remove.append(node)
                else:
                    break
            lineSecRow.insert(LineSection(line_sec))
            for remove_node in to_remove:
                sorted_nodes.remove(remove_node)
            
        return lineSecRow

    @staticmethod
    def lineSecGroup(nodes_dic):
        lineSecGroup = LineSectionGroup()
        for pos in nodes_dic:
            lineSecRow = LineSectionOperator.split_nodes_row(nodes_dic[pos])
            # LineSectionOperator.visualization(lineSecRow.members)
            lineSecGroup.insert(lineSecRow)
        LineSectionOperator.find_neighbors(lineSecGroup,vertical_reverse=False)
        return lineSecGroup
 
        
                
    @staticmethod
    def find_neighbors(lineSecGroup,vertical_reverse=False):
        i = 0
        while i < len(lineSecGroup.members)-1:
            for lower_member in lineSecGroup.members[i].members:
                for higher_member in lineSecGroup.members[i+1].members:
                    [lower_min,lower_max] = lower_member.range
                    [higher_min,higher_max] = higher_member.range
                    if not (lower_min >higher_max or lower_max <higher_min):
                        if not vertical_reverse :
                            lower_member.neighbors.append(higher_member)
                            sorted(lower_member.neighbors,key=lambda x:x.range[0])   
                        else:
                            higher_member.neighbors.append(lower_member)
                            sorted(higher_member.neighbors,key=lambda x:x.range[0])
            i += 1
                    
    
    def decompose(self):
        nodes_dic = self.build_dic()
        LineSecGroup = LineSectionOperator.lineSecGroup(nodes_dic)
        map = []
        for row in LineSecGroup.members:
            for section in row.members:
                group = []
                LineSectionOperator.zig_zag_search(section,group)
                if group:
                    map.append(group)
        return map

        

    @staticmethod
    def zig_zag_search(section,group):
        if not section.visited:
            group.append(section)
            section.visited = True
            if section.neighbors:
                for neighbor in section.neighbors:
                    if not neighbor.visited:
                        LineSectionOperator.zig_zag_search(neighbor,group)
                        break
    
    @staticmethod
    def visualization(map):
        line_set_group = []
        cubes = []
        for group in map:
        # group = map[2]
        # for section in group:
        #     start_point = np.array([section.range[0], section.pos, 4.3], dtype=np.float32)
        #     end_point = np.array([section.range[1], section.pos, 4.3], dtype=np.float32)
        #     line_set = o3d.geometry.LineSet()
        #     line_set.points = o3d.utility.Vector3dVector([start_point, end_point])
        #     line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        #     line_set_group.append(line_set)
        # o3d.visualization.draw_geometries(line_set_group, window_name="Line Set Visualization", width=800, height=600)
            color = list(np.random.rand(3))
            for section in group:

                for node in section.nodes:
                    # 计算每个正方体的中心位置
                    center = np.array(node.center)
                    # 使用 create_mesh_box 函数生成正方体
                    cube = o3d.geometry.TriangleMesh.create_box(width=node.size, height=node.size, depth=node.size)
                    # 平移正方体到指定位置
                    cube.translate(center)
                    cube.paint_uniform_color(color)

                    # 将生成的正方体添加到列表中
                    cubes.append(cube)
        o3d.visualization.draw_geometries(cubes, window_name="Cubes Visualization", width=800, height=600)



if __name__ == "__main__":
    class node:
        def __init__(self,node_info):
            self.center = node_info["center"]
            self.size = node_info["size"]
    node_list = []
    # 读取 Excel 文件
    df = pd.read_excel('B:\Master arbeit\layer_data\layer11.xlsx', sheet_name='Sheet_1')
    for index, row in df.iterrows():
        node_info = {"center":[row["center_x"],row["center_Y"],row["center_Z"]],
                     "size":row["size"]
                    }
        node_list.append(node(node_info))
    test = LineSectionOperator(node_list)
    map = test.decompose()
    LineSectionOperator.visualization(map)
    

