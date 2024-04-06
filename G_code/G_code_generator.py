from cnc import gcode
import queue
from G_code.LineSection import *
class G_code():
    def __init__(self) -> None:
        self.gcode = queue.Queue()
        self.laser_mark = False

    def __G01__(self,point):
        
        x,y= point[0],point[1]
        self.gcode.put(f"G1 X{x} Y{y}")

    def __laserOff__(self):
        self.laser_mark = False
        self.gcode.put(f"M161")

    def __laserOn__(self):
        self.laser_mark = True
        self.gcode.put(f"M160")

    def __G00__(self,point):
        x,y= point[0],point[1]
        self.gcode.put(f"G0 X{x} Y{y}")

    def __paraInit__(self,feed_rate,height):
        F = feed_rate
        z = height
        self.gcode.put(f"G1 F{F}")
        self.gcode.put(f"G0 Z{z}")
        self.__toolSwitching__()

    def __printing__(self,point):
        if not self.laser_mark:
            self.__laserOn__()
        self.__G01__(point)
    
    def __retraction__(self,point):
        if self.laser_mark:
            self.__laserOff__()
        self.__G00__(point)
    
    def __toolSwitching__(self):
        self.gcode.put('G4 P50')
    
    def __save__(self,file_path):
        with open(file_path, 'a') as f:
            while not self.gcode.empty():
                f.write(self.gcode.get()+'\n')

    def main(self,path,file_path):
        # self.__paraInit__(feed_rate=20)
        for single_path in path:
            point_1 = [single_path[0][0],single_path[1][0]]
            point_2 = [single_path[0][1],single_path[1][0]]
            point_3 = [single_path[0][1],single_path[1][1]]
            self.__retraction__(point_1)
            self.__printing__(point_2)
            self.__printing__(point_3)
        self.__save__(file_path)


if __name__ == '__main__':
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
    results,_= tsp.tsp_solver_ot()
    new_path = []
    for index,result in enumerate(results):
        if index%2==0:
            if result[0]>result[1] :
                new_path.append(test.path_reverse(total_path[result[0]//2]))
            else:
                new_path.append(total_path[result[0]//2])
    file_path = 'B:\Master arbeit\layer_data\gcode.txt'
    gcode = G_code()
    with open(file_path, 'w') as f:
        f.write('')
    gcode.__paraInit__(feed_rate=0.2,height = 0.1)
    test.path_reverse(total_path[3])

    for path in new_path:
        gcode.main(path,file_path)

    # tsp = Tsp_path(total_path)
    # # tsp.tsp_solver_ot()
    # LineSectionOperator.visualization(map,total_path,TSP=True)