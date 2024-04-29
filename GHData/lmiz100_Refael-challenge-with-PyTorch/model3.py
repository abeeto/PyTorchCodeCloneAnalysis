# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:02:21 2019

@author: lior_
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim



# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).unsqueeze(1).expand_as(out))
  
    return out


# Initializing the weights of the neural network in an optimal way for the learning (good exploration vs exploitation management)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        
        
class Network(torch.nn.Module):
    
    def __init__(self, num_inputs, num_outputs):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 32) 
        #for learn complex temporal relationships
        self.lstm = nn.LSTMCell(32, 16) 
        self.actor_linear = nn.Linear(16, num_outputs) #actor outputs = Q(s,a)
        self.critic_linear = nn.Linear(16, 1) # critic output = V(s)
        self.apply(weights_init) #initializing random weights by apply weights_init on our model
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.fc1.bias.data.fill_(0)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train() #setting the module in "train" mode to activate the dropouts and batchnorms
        
        
    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        
        x = F.elu(self.fc1(inputs))
        hx, cx = self.lstm(x, (hx, cx))
        x = hx #only the hidden layers are useful
        
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

    
    
class Ai3():
    def __init__(self, params):
        self.brain = Network(5, 4)
        self.optimizer = optim.Adam(self.brain.parameters(), lr = params.lr)
        self.params = params
        self.score_list = []
        
        
    def select_max_action(self, state_vals):
        value, action_value, (self.hx, self.cx) = self.brain((Variable(torch.Tensor(state_vals).float().unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(action_value, dim=1)
        action = prob.max(1)[1].data
        
        return action[0]
    
    
    def select_softmax_action(self, state_vals):
        value, action_value, (self.hx, self.cx) = self.brain((Variable(torch.Tensor(state_vals).float().unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(action_value, dim=1)
        action = prob.multinomial(1)
        
        return action[0,0]
    
    
    def update_game_score(self, score):
         #reset hx, cx before start new game
        self.cx = torch.zeros(1, 16)
        self.hx = torch.zeros(1, 16)
        self.score_list.append(score)
        
        
    def get_avg_score(self):
        return np.mean(self.score_list)