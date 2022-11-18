import torch
from torch import nn
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
class Critic1(nn.Module):
    def __init__(self,num_input,num_action,num_hidden):
        super(Critic1, self).__init__()
        self.linear1 = nn.Linear(in_features=num_input + num_action, out_features=num_hidden)
        self.linear2 = nn.Linear(in_features=num_hidden, out_features=num_hidden)
        self.linear3 = nn.Linear(in_features=num_hidden, out_features=1)
    def forward(self,state,action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x
class Critic2(nn.Module):
    def __init__(self,num_input,num_action,num_hidden):
        super(Critic2, self).__init__()
        self.linear1 = nn.Linear(in_features=num_input + num_action, out_features=num_hidden)
        self.linear2 = nn.Linear(in_features=num_hidden, out_features=num_hidden)
        self.linear3 = nn.Linear(in_features=num_hidden, out_features=1)
    def forward(self,state,action):
        x = torch.cat([state, action], 1)

        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x