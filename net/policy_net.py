import torch
from torch import nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
class Policy_net(nn.Module):
    def __init__(self,num_input,num_action,num_hidden):
        super(Policy_net, self).__init__()
        self.linear1=nn.Linear(in_features=num_input,out_features=num_hidden)
        self.linear2=nn.Linear(in_features=num_hidden,out_features=num_hidden)
        self.linear3=nn.Linear(in_features=num_hidden,out_features=num_action)
    def forward(self,state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        x =torch.tanh(self.linear3(x))
        return x
    def get_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]
