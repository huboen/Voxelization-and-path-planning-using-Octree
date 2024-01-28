import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Env():
    def __init__(self,map) -> None:
        self.map = map
        pass

    def __distance_matrix__(self):
        pass

    def update_status(self):
        pass

    def draw_path(self):
        pass

    def path_cal(self):
        pass

    def state_init(self):
        pass

