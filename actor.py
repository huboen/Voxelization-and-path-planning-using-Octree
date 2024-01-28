import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,input_size,output_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc1.weight.data.normal(0,0.1)
        self.fc2 = nn.Linear(64,64)
        self.fc2.weight.data.normal(0,0.1)
        self.fc3 = nn.Linear(64,64)
        self.fc3.weight.data.normal(0,0.1)
        self.out = nn.linear(64,output_size)

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