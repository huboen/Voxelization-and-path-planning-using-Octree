"""
基于TSP设计的强化学习模拟环境
"""
import sys
import numpy as np
import math
from Transform import Transform
import matplotlib.pyplot as plt
class Env(object):
    def __init__(self,Map,extend=0.2):
        """
        :param Map: 这个Map就是我们城市的一个矩阵,是表示位置的一个矩阵
        """
        self.Map = Map
        self.city_num = len(self.Map)
        self.__matrix_distance = self.__matrix_dis()
        self.city_node = [node for node in range(self.city_num)]
        self.best_dist = float("inf")
        self.best_path = None
        self.transform = Transform()
        self.extend = int(self.city_num*extend)
        self.tolerate_threshold = 0
        self.tanh = math.tanh

    def __matrix_dis(self):
        res = np.zeros((self.city_num, self.city_num))
        for i in range(self.city_num):
            for j in range(i + 1, self.city_num):
                res[i, j] = np.linalg.norm(self.Map[i, :] - self.Map[j, :])
                res[j, i] = res[i, j]
        return res

    def draw_path(self,path):

        ## 绘制初始化的路径图
        fig, ax = plt.subplots()
        x = self.Map[:, 0]
        y = self.Map[:, 1]
        ax.scatter(x, y, linewidths=0.1)
        for i, txt in enumerate(range(1, len(self.Map) + 1)):
            ax.annotate(txt, (x[i], y[i]))
        #获取头结点
        ax.cla()
        res0 = path
        x0 = x[res0]
        y0 = y[res0]
        for i in range(len(self.Map) - 1):
            plt.quiver(x0[i], y0[i], x0[i + 1] - x0[i], y0[i + 1] - y0[i], color='r', width=0.005, angles='xy', scale=1,
                       scale_units='xy')
        plt.quiver(x0[-1], y0[-1], x0[0] - x0[-1], y0[0] - y0[-1], color='r', width=0.005, angles='xy', scale=1,
                   scale_units='xy')
        plt.show()
        plt.pause(0.1)

    def comp_fit(self, one_path):
        """
        计算，咱们这个路径的长度，例如A-B-C-D
        :param one_path:
        :return:
        """
        res = 0
        for i in range(self.city_num - 1):
            res += self.__matrix_distance[one_path[i], one_path[i + 1]]
        res += self.__matrix_distance[one_path[-1], one_path[0]]
        return res

    def reset(self):
        """
        初始化环境，并且返回当前的状态
        这块主要是将当前的节点的顺序给他还有这个矩阵

        :return:
        """
        max_distance = np.max(self.__matrix_distance)
        status = np.zeros((self.city_num+self.extend, self.city_num))
        status[:self.city_num,:] = (self.__matrix_distance)/(max_distance)
        return status

    def out_path(self, one_path,fitNess):
        """
        输出我们的路径顺序
        :param one_path:
        :return:
        """
        res = str(one_path[0] + 1) + '-->'
        for i in range(1, self.city_num):
            res += str(one_path[i] + 1) + '-->'
        res += str(one_path[0] + 1) + '\n'

        self.draw_path(one_path)
        print("最短路线为：",res)
        print("此时的最短路程是：",fitNess)

    def getRoward(self,status,actions,isShow):
        """
        返回当前环境的下一个动作，以及奖励。
        注意status和actions都是numpy类型的
        :param isShow 表示要不要输出当前最好的一个路径
        :return:
        """
        #计算当前下一个的状态
        probabilitys = self.transform.TransProbability(actions)
        #将actions,status重新转化为numpy类型
        status = status.detach().numpy()
        actions = actions.detach().numpy()
        """
        通过概率生成我们的路径
        """
        path_out = []
        for probability in probabilitys:
            probability /= probability.sum()
            path_out.append(np.random.choice(self.city_node, (actions[0].shape), p=probability,replace=False))
        fits = []
        for path in path_out:
            fits.append(self.comp_fit(path))

        great_fits = np.argsort(fits)[:self.extend]
        great_actions = actions[great_fits,:]
        status[self.city_num:,:]=(great_actions)/(self.city_num-1)
        #计算奖励
        great_dist = fits[great_fits[0]]
        R = self.tanh(self.best_dist-great_dist)
        if(self.best_dist>great_dist):
            self.best_dist = great_dist
            self.best_path = path_out[great_fits[0]]

        if(isShow):
            self.out_path(self.best_path,self.best_dist)
        return status,R

if __name__ == '__main__':
    # print(sys.path)
    data = np.array([16.47, 96.10, 16.47, 94.44, 20.09, 92.54,
                     22.39, 93.37, 25.23, 97.24, 22.00, 96.05, 20.47, 97.02,
                     17.20, 96.29, 16.30, 97.38, 14.05, 98.12, 16.53, 97.38,
                     21.52, 95.59, 19.41, 97.13, 20.09, 92.55]).reshape((14, 2))
    env = Env(data)
    print(env.reset())

