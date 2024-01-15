"""
将TSP所构造的环境进行反馈，转换，送入Actor网络以及将Actor的输出，转化为对应的序列
"""
import math
import torch
class Transform(object):
    def __init__(self):
        self.pow = math.pow

    def TransProbability(self,NetOut):
        """
        :param NetOut: 由神经网络输出的一组结果，我要将其得到一个概率
        注意NetOut是一个tensor
        :return:
        """
        Probability = torch.softmax(NetOut,dim=1)
        return Probability.detach().numpy()

