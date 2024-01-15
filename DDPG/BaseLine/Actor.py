import torch
from torch import nn
import torch.nn.functional as F
class Actor(nn.Module):
    """
    这个是我们的Actor网络
    """
    def __init__(self,state,out):
        super(Actor, self).__init__()
        """
        The network structure is modified
        """
        self.fc1 = nn.Linear(state,64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64,64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(64,64)
        self.fc3.weight.data.normal_(0,0.1)
        self.out = nn.Linear(64,out)

    def forward(self,x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        return x

