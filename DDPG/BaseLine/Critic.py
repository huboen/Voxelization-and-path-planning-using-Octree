from torch import nn
import torch.nn.functional as F
class Critic(nn.Module):
    """
    这个是我们的Critic网络，因为DQN很难去直接表示三维消息
    如果要就需要一个映射表，这个映射表的动作也是很复杂的，还是需要一个Net
    所以的话我们这边还是直接选择使用这个DDPG
    """
    def __init__(self,state_dim,action_dim):
        """
        :param state_action:
        """
        super(Critic,self).__init__()

        self.fc1_status = nn.Linear(state_dim,64)
        self.fc1_status.weight.data.normal_(0,0.1)

        self.fc1_actions = nn.Linear(action_dim,64)
        self.fc1_actions.weight.data.normal_(0,0.1)

        self.fc2_status= nn.Linear(64,32)
        self.fc2_status.weight.data.normal_(0,0.1)

        self.fc2_actions = nn.Linear(64,32)
        self.fc2_actions.weight.data.normal_(0,0.1)

        self.fc5 = nn.Linear(32,16)
        self.fc5.weight.data.normal_(0,0.1)
        self.out = nn.Linear(16,1)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,status,actions):
        status = self.fc1_status(status)
        status = F.leaky_relu(status)
        status = self.fc2_status(status)

        actions = self.fc1_actions(actions)
        actions = F.leaky_relu(actions)
        actions = self.fc2_actions(actions)
        net = status+actions
        net = F.leaky_relu(net)
        net = self.fc5(net)
        net = F.leaky_relu(net)
        out  = self.out(net)
        return out

