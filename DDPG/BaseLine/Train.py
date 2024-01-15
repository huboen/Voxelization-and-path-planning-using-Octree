from Env import Env
from Actor import Actor
from Critic import Critic
import torch
from torch import nn
import numpy as np
class DDPGTSP(object):
    """
    我们这里的话就直接使用cpu了，就不去使用这个GPU了
    """

    def __init__(self,Map,extend=0.2,
                 memory_capacity_critic=20,lr_actor=0.01,lr_critic=0.01,
                 epsilon=0.9,gamma=0.9,batch_size_critic=10,
                 target_replace_iter=5
                 ):
        self.Map = Map
        self.extend = extend
        self.env = Env(self.Map,extend=self.extend)
        self.memory_capacity_critic = memory_capacity_critic
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size_critic = batch_size_critic
        self.target_replace_iter = target_replace_iter
        """
        创建网络，定义优化器，损失函数等等
        """
        self.status_dim = self.actions_dim = self.env.city_num
        self.actor_eval,self.actor_target = Actor(self.status_dim,self.status_dim),\
                                            Actor(self.status_dim, self.status_dim)
        self.eval_net_critic, self.target_net_critic = Critic(self.status_dim,self.actions_dim),\
                                                       Critic(self.status_dim,self.actions_dim)

        self.opt_actor = torch.optim.Adam(self.actor_eval.parameters(), lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(self.eval_net_critic.parameters(), lr=self.lr_critic)

        self.learn_step_count = 0
        self.memory_count = 0
        """
        由于我们输出的就是二维的，所以我们的记忆库是三维的
        （当然也可以选择直接打平变成二维的），但是这里是3维的
        s,a,r,s_
        """
        self.memory = np.zeros((self.memory_capacity_critic, self.env.extend+self.env.city_num,
                                self.status_dim + self.actions_dim + self.status_dim + 1)
                               )

        self.loss_func_critic = nn.MSELoss()

    def remember(self, s, a, r, s_):
        """
        存储记忆
        :param s: 当前的状态
        :param a: 当前状态对应的动作组
        :param r: 当前的获得的奖励
        :param s_:下一个时刻的状态
        :return:
        """


        transition = np.hstack((s, a.detach().numpy(), [[r] for _ in range(a.shape[0])], s_))
        index = self.memory_count % self.memory_capacity_critic
        self.memory[index, :] = transition
        self.memory_count += 1

    def savaMould(self, net, path):
        """
        :return:
        """
        torch.save(net.state_dict(), path)

    def loadMould(self, path_actor, path_critic):
        """
        :return:
        """
        self.actor_eval.load_state_dict(torch.load(path_actor))
        self.eval_net_critic.load_state_dict(torch.load(path_critic))

    def loadMouldActor(self, path_actor):
        """
        :return:
        """
        self.actor_eval.load_state_dict(torch.load(path_actor))

    def loadMouldCritic(self, path_critic):
        """
        :return:
        """
        self.eval_net_critic.load_state_dict(torch.load(path_critic))

    def train_cirtic(self):
        """
        负责训练我们的critic评分网络
        :return:
        """
        self.learn_step_count += 1
        if self.learn_step_count % self.target_replace_iter == 0:
            self.target_net_critic.load_state_dict(self.eval_net_critic.state_dict())
            self.actor_target.load_state_dict(self.actor_eval.state_dict())

        SelectMemory = np.random.choice(self.memory_capacity_critic, self.batch_size_critic)
        selectM = self.memory[SelectMemory, :]
        S_s = torch.FloatTensor(selectM[:,:,:self.status_dim])
        S_a = torch.FloatTensor(selectM[:,:,self.status_dim:self.status_dim + self.actions_dim].astype(int))
        S_r = torch.FloatTensor(selectM[:,:,self.status_dim + self.actions_dim:self.status_dim + self.actions_dim + 1])
        S_s_ = torch.FloatTensor(selectM[:,:,-self.status_dim:])


        q_eval = self.eval_net_critic(S_s,S_a)

        S_a_ = self.actor_target(S_s_)

        b = torch.normal(mean=torch.full((self.batch_size_critic,self.env.extend+self.env.city_num,self.status_dim), 0.0),
                         std=torch.full((self.batch_size_critic,self.env.extend+self.env.city_num,self.status_dim), 0.5))
        S_a_ = S_a_ + b


        q_next = self.target_net_critic(S_s_,S_a_).detach()
        q_target = S_r + self.gamma * q_next.max(-1)[0].view(self.batch_size_critic,self.env.extend+self.env.city_num,1)
        loss = self.loss_func_critic(q_eval, q_target)

        self.opt_critic.zero_grad()
        loss.backward()
        self.opt_critic.step()

    def train_actor(self,status_tensor)->torch.tensor:
        """
        复杂对actor网络进行训练
        :return:
        """
        out_actor = self.actor_eval(status_tensor)
        loss = -torch.mean(self.eval_net_critic(status_tensor,out_actor))
        self.opt_actor.zero_grad()
        loss.backward()
        self.opt_actor.step()
        return out_actor

def trainTsp(data,actor_path="b:/Master arbeit/DDPG/BaseLine/actor.pth",critic_path="b:/Master arbeit/DDPG/BaseLine/critic.pth",
                 epoch=10,iteration=1000,
                 show_iter = 200
                 ):
        """
        完成对我们的一个DDPG的TSP问题的一个训练求解
        这里主要就是两件事情
        1.完成我们的一个训练
        2.得到一个模型
        :return:
        """
        ddpgTsp = DDPGTSP(data)
        for epoch in range(epoch):
            status = ddpgTsp.env.reset()
            for iter in range(iteration):
                isShow = False
                if((iter+1)%show_iter==0):
                    """
                    每30次我们显示一下
                    """
                    print("No:", epoch, "in epoch No:", (iter + 1), "times")
                    isShow = True
                status_tensor = torch.tensor(status, dtype=torch.float, requires_grad=True)
                out_action = ddpgTsp.train_actor(status_tensor)
                next_status,reward = ddpgTsp.env.getRoward(status_tensor,out_action,isShow)
                ddpgTsp.remember(status, out_action, reward, next_status)
                if (ddpgTsp.memory_count > ddpgTsp.memory_capacity_critic):
                    ddpgTsp.train_cirtic()
                status = next_status
            print("No",epoch,"-->the best way is:", ddpgTsp.env.best_dist)
            print("No",epoch,"-->the best way is:", ddpgTsp.env.best_path)
        ddpgTsp.savaMould(ddpgTsp.actor_eval, actor_path)
        ddpgTsp.savaMould(ddpgTsp.eval_net_critic, critic_path)
        print("Task completed, the model has been saved. ")

if __name__ == '__main__':
    data = np.array([16.47, 96.10, 16.47, 94.44, 20.09, 92.54,
                     22.39, 93.37, 25.23, 97.24, 22.00, 96.05, 20.47, 97.02,
                     17.20, 96.29, 16.30, 97.38, 14.05, 98.12, 16.53, 97.38,
                     21.52, 95.59, 19.41, 97.13, 20.09, 92.55]).reshape((14, 2))
    ddpgTsp = DDPGTSP(data)
    trainTsp(ddpgTsp.Map,epoch=100,iteration=300,show_iter=300)

