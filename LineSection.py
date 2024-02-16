import pandas as pd
import time



class LineSection:
    def __init__(self,input) -> None:
        self.nodes = input["nodes"]
        self.range = input["range"]
        self.pos = input["y_position"]
        self.neighbors = []


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
        return nodes_dic

    @staticmethod
    def split_nodes_row(nodes):
        lineSecRow = LineSectionRow(nodes[0].center[1])
        sorted_nodes = sorted(nodes,key = lambda x:x.center[0])
        while sorted_nodes:
            line_sec = {"nodes":[],
                        "range":[float('-inf'),float('-inf')],
                        "y_position":nodes[0].center[1]}
            prev_node = None
            for node in sorted_nodes:
                if prev_node is None or node.center[0] == prev_node.center[0]+prev_node.size:
                    line_sec["nodes"].append(node)
                    line_sec["range"] = [line_sec["nodes"][0].center[0],line_sec["nodes"][-1].center[0]]
                    sorted_nodes.remove(node)
                prev_node = node
            lineSecRow.insert(LineSection(line_sec))
        return lineSecRow

    @staticmethod
    def lineSecGroup(nodes_dic):
        lineSecGroup = LineSectionGroup()
        for pos in nodes_dic:
            lineSecRow = LineSectionOperator.split_nodes_row(nodes_dic[pos])
            lineSecGroup.insert(lineSecRow)
        LineSectionOperator.find_neighbors(lineSecGroup,vertical_reverse=False)
 
        
                
    @staticmethod
    def find_neighbors(lineSecGroup,vertical_reverse=False):
        i = 0
        while i < len(lineSecGroup.members)-1:
            for lower_member in lineSecGroup.members[i].members:
                for higher_member in lineSecGroup.members[i+1].members:
                    [lower_min,lower_max] = lower_member.range
                    [higher_min,higher_max] = higher_member.range
                    if lower_min >higher_max or lower_max <higher_min:
                        if not vertical_reverse :
                            lower_member.neighbors.append(higher_member)
                            sorted(lower_member.neighbors,key=lambda x:x.range[0])   
                        else:
                            higher_member.neighbors.append(lower_member)
                            sorted(higher_member.neighbors,key=lambda x:x.range[0])
            i += 1
                    
    
    def decompose(self):
        nodes_dic = self.build_dic()
        
        LineSectionOperator.lineSecGroup(nodes_dic)
        print("finish")
    
if __name__ == "__main__":
    class node:
        def __init__(self,node_info):
            self.center = node_info["center"]
            self.size = node_info["size"]
    node_list = []
    # 读取 Excel 文件
    df = pd.read_excel('B:\Master arbeit\layer_data\layer11.xlsx', sheet_name='Sheet_1')
    for index, row in df.iterrows():
        node_info = {"center":[row["center_x"],row["center_Y"]],
                     "size":row["size"]
                    }
        node_list.append(node(node_info))
    test = LineSectionOperator(node_list)
    test.decompose()
    

