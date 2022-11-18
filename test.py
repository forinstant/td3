#参考TD3原理食用：https://zhuanlan.zhihu.com/p/55307499
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
#在连续动作空间中，策略变化缓慢，current Q与target Q变化不大，
#所以TD3还是沿用Double DQN之前的Double Q-learning的思想，使用两个独立的Critic来防止过估计。
#同时为了防止高方差（variance），又在其基础上提出了clipped Double Q-learning以及Delayed Policy Updates用于均衡。

#记忆库
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
#保存训练记忆
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
#随机抽样
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

#gym.action打包成连续动作
class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

#添加噪声进行探索
class GaussianExploration(object):
    def __init__(self, action_space, max_sigma=1.0, min_sigma=1.0, decay_period=1000000):
        self.low = action_space.low
        self.high = action_space.high
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def get_action(self, action, t=0):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        action = action + np.random.normal(size=len(action)) * sigma
        return np.clip(action, self.low, self.high)

#TD3中使用的第二个技巧就是对Policy进行延时更新，即使用target network。
#target network与critic并不同步更新，这样一来就可以减少之前我们提到的累计误差，从而降低方差。
def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

#画图
def plot(frame_idx, rewards):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

#用神经网络来当作Q表=价值网络
class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

#策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        #均匀分布
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
#前向传播
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x
#输入state,输出action。
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0,0]

class TD(object):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(TD, self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 128
        self.gamma = 0.99
        self.soft_tau = 1e-2
        self.noise_std = 0.2
        self.noise_clip = 0.5
        self.policy_update = 2
        self.soft_tau = 1e-2
        #soft更新
        self.replay_buffer_size = 1000000
        #记忆库的容量
        self.value_lr = 1e-3
        #价值网络的学习率
        self.policy_lr = 1e-3
        #策略网络的学习率
        #把数据丢到GPU计算，因为GPU擅长并行计算和浮点计算
        self.value_net1 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.value_net2 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.target_value_net1 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_value_net2 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        soft_update(self.value_net1, self.target_value_net1, soft_tau=1.0)
        soft_update(self.value_net2, self.target_value_net2, soft_tau=1.0)
        soft_update(self.policy_net, self.target_policy_net, soft_tau=1.0)

        self.value_criterion = nn.MSELoss()
        #平均平方误差，简称均方误差，均方误差是指参数估计值与参数真值之差平方的期望值，记为MSE。
        #MSE是衡量“平均误差”的一种较为方便的方法，MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。

        self.value_optimizer1 = optim.Adam(self.value_net1.parameters(), lr=self.value_lr)
        self.value_optimizer2 = optim.Adam(self.value_net2.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        #Adam吸收了Adagrad（自适应学习率的梯度下降算法）和动量梯度下降算法的优点，既能适应稀疏梯度（即自然语言和计算机视觉问题），
        #又能缓解梯度震荡的问题
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)


    def td3_update(self, step, batch_size):
        #从记忆库读取数据准备训练
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        next_action = self.target_policy_net(next_state)
        #next_state 输入到目标策略网络输出next_action
        noise = torch.normal(torch.zeros(next_action.size()), self.noise_std).to(device)
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        #添加噪声给next_action,权衡探索和利用
        next_action += noise

        target_q_value1  = self.target_value_net1(next_state, next_action)
        #输入状态和动作，输出值函数
        target_q_value2  = self.target_value_net2(next_state, next_action)
        target_q_value   = torch.min(target_q_value1, target_q_value2)
        expected_q_value = reward + (1.0 - done) * self.gamma * target_q_value

        q_value1 = self.value_net1(state, action)
        q_value2 = self.value_net2(state, action)

        value_loss1 = self.value_criterion(q_value1, expected_q_value.detach())#value-loss1
        value_loss2 = self.value_criterion(q_value2, expected_q_value.detach())#value-loss2
        #初始化
        self.value_optimizer1.zero_grad()
        value_loss1.backward()
        self.value_optimizer1.step()

        self.value_optimizer2.zero_grad()
        value_loss2.backward()
        self.value_optimizer2.step()
        #step/sel.policy_update=0 参数更新
        if step % self.policy_update == 0:
            policy_loss = self.value_net1(state, self.policy_net(state))
            policy_loss = -policy_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            soft_update(self.value_net1, self.target_value_net1, soft_tau=self.soft_tau)
            #value_net1参数更新
            soft_update(self.value_net2, self.target_value_net2, soft_tau=self.soft_tau)
            #value_net2参数更新
            soft_update(self.policy_net, self.target_policy_net, soft_tau=self.soft_tau)
            #policy_net参数更新
#训练部分
def main():
    env = NormalizedActions(gym.make('Pendulum-v1'))#环境加载

    noise = GaussianExploration(env.action_space)#噪声动作输出

    state_dim = env.observation_space.shape[0]#状态
    action_dim = env.action_space.shape[0]#动作
    hidden_dim = 256#隐藏层

    TD3 = TD(action_dim, state_dim, hidden_dim)#输入神经网络


    max_frames = 10000
    #最大回合数4
    max_steps = 500
    #回合内最大步长
    frame_idx = 0
    #回合从0开始
    rewards = []
    #数组储存每回合的rewards，用来绘图
    batch_size = 128
    #批次训练大小
# 当在回合内，训练继续
    while frame_idx < max_frames:
        state = env.reset()
        #重置环境
        episode_reward = 0
        #回合奖励重置

        for step in range(max_steps):
            #step<500
            action = TD3.policy_net.get_action(state)
            #获取动作
            #action = noise.get_action(action, step)
            #添加噪声
            next_state, reward, done, _ = env.step(action)
            #赋值给next_state,reward,done

            TD3.replay_buffer.push(state, action, reward, next_state, done)
            #储存数据到记忆库
            if len(TD3.replay_buffer) > batch_size:
                #当记忆库大小大于批次训练大小
                TD3.td3_update(step, batch_size)
                #开始训练参数
            state = next_state
            #更新状态
            episode_reward += reward
            #回合奖励累加
            frame_idx += 1
            #回合结束，回合加1

            if frame_idx % 1000 == 0:
                #回合/1000==0
                plot(frame_idx, rewards)
                #绘制回合/奖励图
            if done:
                #结束
                break
                #跳出
        rewards.append(episode_reward)
        #回合总rewards

if __name__ == '__main__':
    main()





