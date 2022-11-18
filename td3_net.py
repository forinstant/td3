import copy
import os
import torch
import numpy as np
from TD3.util.Replay_Buffer import ReplayBuffer
from TD3.net.policy_net import Policy_net
from TD3.net.critic_net import Critic1
from TD3.net.critic_net import Critic2

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
class TD3(object):
    def __init__(self,action_dim, state_dim, hidden_dim=256):
        super(TD3, self).__init__()
        self.tau=5e-3# 软更新频率
        self.global_step = 0
        self.update_target_step = 2#更新actor网络频率
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 128
        self.gamma = 0.9
        self.replay_buffer_size = 5000#经验池容量
        self.rb = ReplayBuffer(self.replay_buffer_size)#初始化经验池
        self.replay_startsize = 3000#经验池启动阈值
        self.critic_lr = 1e-3
        self.actor_lr = 1e-4
        self.noise_std=0.2#target_action标准差
        self.noise_clip=0.5#噪声范围
        self.actor_net=Policy_net(num_input=self.state_dim,num_action=self.action_dim,num_hidden=self.hidden_dim).to(device)
        self.target_actor_net=copy.deepcopy(self.actor_net).to(device)
        self.critic_net1=Critic1(num_input=self.state_dim,num_action=self.action_dim,num_hidden=self.hidden_dim).to(device)
        self.target_critic_net1=copy.deepcopy(self.critic_net1)
        self.critic_net2=Critic2(num_input=self.state_dim,num_action=self.action_dim,num_hidden=self.hidden_dim).to(device)
        self.target_critic_net2 = copy.deepcopy(self.critic_net2)

        self.critic1_optimizer = torch.optim.Adam(self.critic_net1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic_net2.parameters(), lr=self.critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_loss1 = torch.nn.MSELoss()
        self.critic_loss2 = torch.nn.MSELoss()
        self.min_value = -np.inf#防止数据爆炸
        self.max_value = np.inf#防止数据爆炸

    def learn(self,state, action, reward, next_state, done):
        self.global_step += 1
        self.rb.push(state, action, reward, next_state, done)#经验池添加经验
        if len(self.rb) >= self.replay_startsize:
            self.learn_batch(*self.rb.sample(self.batch_size))#网络学习
    def learn_batch(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done):
        batch_next_action=self.target_actor_net(batch_next_state)
        noise = torch.normal(torch.zeros(batch_next_action.size()), self.noise_std).to(device)# 均值为0 标准差为self.noise_std正态分布噪声
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)#限制噪声大小
        batch_next_action+=noise#添加噪声

        target_Qvalue1=self.target_critic_net1(batch_next_state, batch_next_action.detach())
        target_Qvalue2=self.target_critic_net2(batch_next_state, batch_next_action.detach())
        target_Qvalue=torch.min( target_Qvalue1,target_Qvalue2)
        expected_q_value = batch_reward + (1.0 - batch_done) * self.gamma * target_Qvalue
        expected_value = torch.clamp(expected_q_value, self.min_value, self.max_value)
        Qvalue1=self.critic_net1(batch_state,batch_action)
        Qvalue2=self.critic_net2(batch_state,batch_action)
        value_loss1=self.critic_loss1(Qvalue1,expected_value.detach())
        value_loss2=self.critic_loss2(Qvalue2,expected_value.detach())
        self.critic1_optimizer.zero_grad()
        value_loss1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        value_loss2.backward()
        self.critic2_optimizer.step()
        if self.global_step%self.update_target_step==0:
            actor_loss=self.critic_net1(batch_state,self.actor_net(batch_state))
            actor_loss=-actor_loss.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()#更新actor网络  每更新两次critic更新一次actor 网络
            self.soft_update(net=self.actor_net,target_net=self.target_actor_net)#软更新target网络
            self.soft_update(net=self.critic_net1, target_net=self.target_critic_net1)#软更新target网络
            self.soft_update(net=self.critic_net2, target_net=self.target_critic_net2)#软更新target网络


    def soft_update(self,net, target_net):
        for target_parma, parma in zip(target_net.parameters(), net.parameters()):
            target_parma.data.copy_(
                target_parma.data * (1.0 - self.tau) + parma.data * self.tau
            )
    def save(self,eposide,reward):
        torch.save(self.critic_net2.state_dict(),
                   os.path.join('../critic2_pth', "eposide=%d   reward=%d.pth" % (eposide, reward)))
        torch.save(self.critic_net1.state_dict(),
                   os.path.join('../critic1_pth', "eposide=%d   reward=%d.pth" % (eposide, reward)))
        torch.save(self.actor_net.state_dict(),
                   os.path.join('../actor_pth', "eposide=%d   reward=%d.pth" % (eposide, reward)))
    def load(self,):

            self.critic_net1.load_state_dict(torch.load(''))
            self.actor_net.load_state_dict(torch.load(''))
            self.critic_net2.load_state_dict(torch.load(''))

