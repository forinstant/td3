import collections
import random
from torch import FloatTensor
import torch
import numpy as np
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
'''class ReplayBuffer():
    def __init__(self,max_size):
        self.buffer=collections.deque(maxlen=max_size)

    def append(self,exp):
        self.buffer.append(exp)


    def sample(self,batch_size):
        mini_batch=random.sample(self.buffer,batch_size)

        obs_batch,action_batch,reward_batch,next_obs_batch,done_batch=map(np.stack, zip(*mini_batch))
        obs_batch=FloatTensor(obs_batch).to(device)
        action_batch=FloatTensor(action_batch).to(device)
        reward_batch=FloatTensor(reward_batch).unsqueeze(1).to(device)
        next_obs_batch=FloatTensor(next_obs_batch).to(device)
        done_batch=FloatTensor(done_batch).unsqueeze(1).to(device)
        return obs_batch,action_batch,reward_batch,next_obs_batch,done_batch
    def __len__(self):
        return len(self.buffer)'''
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
#随机抽样
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)

        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = map(np.stack, zip(*mini_batch))

        obs_batch = FloatTensor(obs_batch).to(device)
        action_batch = FloatTensor(action_batch).to(device)

        reward_batch = FloatTensor(reward_batch).unsqueeze(1).to(device)
        next_obs_batch = FloatTensor(next_obs_batch).to(device)
        done_batch = FloatTensor(done_batch).unsqueeze(1).to(device)
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)
