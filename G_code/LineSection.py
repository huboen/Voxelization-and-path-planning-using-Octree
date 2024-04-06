import pandas as pd
import time
import open3d as o3d
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import os
import glob
from python_tsp.exact import solve_tsp_dynamic_programming
import pulp
from pulp import LpProblem, LpMinimize, PULP_CBC_CMD
from G_code.test_LP import main
import copy





class node:
    def __init__(self,node_info):
        self.center = node_info["center"]
        self.size = node_info["size"]



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
    def lineSecGroup(nodes_dic,vertical_reverse = False):
        lineSecGroup = LineSectionGroup()
        for pos in nodes_dic:
            lineSecRow = LineSectionOperator.split_nodes_row(nodes_dic[pos])
            # LineSectionOperator.visualization(lineSecRow.members)
            lineSecGroup.insert(lineSecRow)
        LineSectionOperator.find_neighbors(lineSecGroup,vertical_reverse=vertical_reverse)
        return lineSecGroup
 
        
                
    @staticmethod
    def find_neighbors(lineSecGroup,vertical_reverse=False):
        size = lineSecGroup.members[0].members[0].nodes[0].size
        i = 0
        while i < len(lineSecGroup.members)-1:
            for lower_member in lineSecGroup.members[i].members:
                for higher_member in lineSecGroup.members[i+1].members:
                    [lower_min,lower_max] = lower_member.range
                    [higher_min,higher_max] = higher_member.range
                    if not (lower_min-size >higher_max or lower_max+size <higher_min):
                        if not vertical_reverse :
                            lower_member.neighbors.append(higher_member)
                            sorted(lower_member.neighbors,key=lambda x:x.range[0])   
                        else:
                            higher_member.neighbors.append(lower_member)
                            sorted(higher_member.neighbors,key=lambda x:x.range[0])
            i += 1
                    
    
    def decompose(self,vertical_reverse = False):
        nodes_dic = self.build_dic()
        LineSecGroup = LineSectionOperator.lineSecGroup(nodes_dic,vertical_reverse=vertical_reverse)
        map = []
        if not vertical_reverse:
            for row in LineSecGroup.members:
                for section in row.members:
                    group = []
                    LineSectionOperator.zig_zag_search(section,group)
                    if group:
                        map.append(group)
        else:
            for row in LineSecGroup.members[::-1]:
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
    def path_generator(group,vertical_reverse = False):
        path = []
        direction = 1  
        # direction = 1  right to left
        # direction = 0 left to right
        if not vertical_reverse:
            lineSec_prev = None
        if len(group)-1:
            for index in range(len(group)-1):
                lineSec = group[index]
                if lineSec_prev is None:
                    lineSec_prev = lineSec
                else:
                    lineSec_prev = group[index-1]
                lineSec_next = group[index+1]
                if direction == 1:
                    path.append([[max(lineSec_prev.range[1],lineSec.range[1]),min(lineSec_next.range[0],lineSec.range[0])],[lineSec.pos,lineSec_next.pos]])
                    direction = not direction
                else:
                    path.append([[min(lineSec_prev.range[0],lineSec.range[0]),max(lineSec_next.range[1],lineSec.range[1])],[lineSec.pos,lineSec_next.pos]])
                    direction = not direction

        else:
            lineSec = group[-1]
            path.append([[max(lineSec.range),min(lineSec.range)],[lineSec.pos,lineSec.pos]])

        return path
    
    @staticmethod
    def path_reverse(path):
        path_rev = copy.deepcopy(path[::-1])
        for index in range(len(path_rev)-1):
            path_rev[index][0] = path_rev[index][0][::-1]
            path_rev[index][1][1]= path_rev[index+1][1][0]
        return path_rev


    @staticmethod
    def visualization(map,total_path,TSP=False):
        line_set_group = []
        cubes = []
        for group in map:
            cubes2 = []
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
                    cubes2.append(cube)
            # o3d.visualization.draw_geometries(cubes2, window_name="Cubes Visualization", width=800, height=600)
        o3d.visualization.draw_geometries(cubes, window_name="Cubes Visualization", width=800, height=600)

        for path in total_path:
            color = np.random.rand(3)
            for  single_path in path:
                start = [single_path[0][0],single_path[1][0]]
                end = [single_path[0][1],single_path[1][0]]
                plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], head_width=0.1, head_length=0.1, fc='blue', ec=color)
                if path.index(single_path)+1!=len(path):
                    plt.arrow(end[0], end[1], 0, single_path[1][1]-end[1], head_width=0.1, head_length=0.1, fc='blue', ec=color)

        if TSP:
            tsp_solver = Tsp_path(total_path)
            results,_ = tsp_solver.tsp_solver_ot()
            for result in results:
                end_0,end_1= result[0],result[1]
                if (end_1-end_0==1 and end_0%2 ==0 ) or (end_0-end_1==1 and end_1%2 ==0 ):
                    continue
                try:
                    group1 = total_path[end_0//2]
                    group2 = total_path[end_1//2]
                except Exception as e:
                    print(results)
                    raise ValueError(f"{end_0,end_1}")
                point1 = [group1[-math.ceil(end_0%2)][0][math.ceil(end_0%2)],group1[-math.ceil(end_0%2)][1][0]]
                point2 = [group2[-math.ceil(end_1%2)][0][math.ceil(end_1%2)],group2[-math.ceil(end_1%2)][1][0]]
                plt.arrow(point1[0], point1[1], point2[0]-point1[0], point2[1]-point1[1], head_width=0.1, head_length=0.1, fc='blue', ec="red",linewidth=1)

        plt.xlim(-75, 75)
        plt.ylim(-75, 75)
        plt.gca().set_aspect(1)
        plt.show()
    



class Printing_3d:
    def __init__(self,input) -> None:
        self.input = input
        #input = {"printing_speed in mm/s": 0.2,
        #          "acceleration in mm^2/s" : 0
        #          "Nozzle Change Time in ms": 50      
        #       }
    
    def __printing_speed_control__(self,size):
        print_speed = self.input["printing_speed in mm/s"]

        return print_speed

    def layer_printing_time(self,total_path,size):
        total_time = 0
        printingInfo= self.input
        printingSpeed = self.__printing_speed_control__(size)
        path_solver = Tsp_path(total_path)
        _,distance = path_solver.tsp_solver_ot()
        path_distance = 0
        for single_path in total_path:
            for section in single_path:
                path_distance += max(section[0])-min(section[0])
        total_distance = path_distance + distance
        total_time = total_distance/printingSpeed
        return total_time

    def printing_time(self,model):
        printingInfo= self.input
        nozzleChangeTime = printingInfo["Nozzle Change Time in s"]
        changeLabel = False
        nozzle_size = None
        Totaltime = 0
        for layer in model:
            if nozzle_size is None:
                nozzle_size = layer.size
            else:
                if nozzle_size != layer.size:
                    changeLabel = True
            Totaltime += self.layer_printing_time(layer.total_path,layer.size) + changeLabel * nozzleChangeTime
            changeLabel = False
        return Totaltime
    
    @staticmethod
    def load_model(folder_path):
        xlsx_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        layer_list = []
        for file_path in xlsx_files:
            df = pd.read_excel(file_path, sheet_name='Sheet_1')
            node_list = []
            for index, row in df.iterrows():

                node_info = {"center":[row["center_x"],row["center_Y"],row["center_Z"]],
                            "size":row["size"]
                            }
                node_list.append(node(node_info))
            layer_list.append(node_list)
        sorted_layers = sorted(layer_list,key=lambda x:x[0].center[2])
        return sorted_layers
    

class Tsp_path:
    def __init__(self,total_path) -> None:
        self.total_path = total_path

    def __matrixUpdate__(self):
        n = len(self.total_path) * 2
        position_list = []
        distance_matrix = np.ones([n,n])
        constraints = []
        for index,single_path in enumerate(self.total_path):
            path_distance = 0
            single_path = np.array(single_path)
            end_1 = [single_path[0,0,0],single_path[0,1,0]]
            end_2 = [single_path[-1,0,0],single_path[-1,1,0]]
            position_list.extend([end_1,end_2])
            for section in single_path:
                path_distance += max(section[0])-min(section[0])
            distance_matrix[2*index,2*index+1] = 0
            distance_matrix[2*index+1,2*index] = 0
            constraints.extend([[2*index,2*index+1]])
            
        n = len(position_list)
        
        for index_x,x in enumerate(position_list):
            for index_y,y in enumerate(position_list):
                if distance_matrix[index_x,index_y] == 1:
                    distance_matrix[index_x,index_y] = int(np.linalg.norm(np.array(y)-np.array(x))*100)
        return distance_matrix,constraints

    def tsp_solver(self):
        distance_matrix,constraints = self.__matrixUpdate__()
        n = distance_matrix.shape[0]
        # 创建问题实例
        problem = pulp.LpProblem("TSP Problem", pulp.LpMinimize)

        # 创建决策变量
        x = [[pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary) for j in range(n)] for i in range(n)]
        y = {}
        for index,constraint in enumerate(constraints):
            y[index] = pulp.LpVariable(f"y_{constraint[0]}_{constraint[1]}", cat=pulp.LpBinary)


        # 定义目标函数
        problem += pulp.lpSum(distance_matrix[i][j] * x[i][j] for i in range(n) for j in range(n))

        # 添加约束条件
        # 每个城市只能访问一次
        for i in range(n):
            problem += pulp.lpSum(x[i][j] for j in range(n)) == 1
            problem += pulp.lpSum(x[j][i] for j in range(n)) == 1

        # 确保路径形成一个环路
        for i in range(n):
            problem += x[i][i] == 0
        # 添加线段连接约束
        for index,constraint in enumerate(constraints):
            problem += x[constraint[0]][constraint[1]] == y[index]
            problem += x[constraint[1]][constraint[0]] == 1 - y[index]
        
        # 创建变量表示每个城市的访问顺序
        u = [pulp.LpVariable(f"u_{i}", lowBound=0, upBound=n-1, cat=pulp.LpInteger) for i in range(n)]

        # 添加MTZ约束，确保不会形成子环
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    problem += u[i] - u[j] + n * x[i][j] <= n - 1
        time_limit_in_seconds = 60*2

        problem.solve(PULP_CBC_CMD(msg=1, timeLimit=time_limit_in_seconds))
        result = []
        # print("Optimal path:")
        for i in range(n):
            for j in range(n):
                if pulp.value(x[i][j]) == 1:
                    result.append([i,j])
        #             print(f"City {i} -> City {j}, Distance: {distance_matrix[i][j]}")
        distance = round(pulp.value(problem.objective),3)
        
        return result,distance
    
    def tsp_solver_ot(self):
        distance_matrix,_= self.__matrixUpdate__()
        results,distance = main(distance_matrix)
        new_path = []
        for index,result in enumerate(results):
            if index%2==0:
                if result[0]>result[1] :
                    new_path.append(LineSectionOperator.path_reverse(self.total_path[result[0]//2]))
                else:
                    new_path.append(self.total_path[result[0]//2])
        # print(result)
        return results,new_path
    

if __name__ == "__main__":


    node_list = []
    # 读取 Excel 文件
    df = pd.read_excel('B:\Master arbeit\layer_data\layer11.xlsx', sheet_name='Sheet_1')
    for index, row in df.iterrows():
        node_info = {"center":[row["center_x"],row["center_Y"],row["center_Z"]],
                     "size":row["size"]
                    }
        node_list.append(node(node_info))
    test = LineSectionOperator(node_list)
    map = test.decompose(vertical_reverse=False)
    total_path = []
    for group in map:
        total_path.append(test.path_generator(group))
    tsp = Tsp_path(total_path)
    # tsp.tsp_solver_ot()
    LineSectionOperator.visualization(map,total_path,TSP=True)
    
    # input = {"printing_speed in mm/s": 50,
    #         "acceleration in mm^2/s" : 0,
    #         "Nozzle Change Time in s": 0.05      
    #         }
    # printer = Printing_3d(input)
    # print_time = printer.layer_printing_time(total_path,node_list[0].size)
    # print(print_time)


    

