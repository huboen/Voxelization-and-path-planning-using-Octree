from G_code.G_code_generator import *
def g_code_create(folder_path,feed_rate=1):
    layer_list = Printing_3d.load_model(folder_path)
    print("loaded")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir,"Path", "Path_gcode.nc")
    # file_path = 'B:\Master arbeit\layer_data\gcode_all.nc'
    with open(file_path, 'w') as f:
        f.write('')
    for layer in layer_list:
        gcode = G_code()
        test = LineSectionOperator(layer)
        map = test.decompose(vertical_reverse=False)
        total_path = []
        for group in map:
            # if test.path_generator(group) is []:
            #     print(group)
            #     raise ValueError
            total_path.append(test.path_generator(group))
            if [] in total_path:
                print(group)
                print(test.path_generator(group))
                raise ValueError
        tsp = Tsp_path(total_path)
        results,new_path= tsp.tsp_solver_ot()
        gcode.__paraInit__(feed_rate=feed_rate,height=layer[0].center[2])
        for path in new_path:
            gcode.main(path,file_path)

if __name__ == '__main__': 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir,"layer_data")          
    # folder_path = 'B:\Master arbeit\layer_data'
    g_code_create(folder_path=folder_path,feed_rate=1)
