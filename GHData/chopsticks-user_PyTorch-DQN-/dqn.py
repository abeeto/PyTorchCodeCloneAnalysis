import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from torch.optim import lr_scheduler as ls
from collections import namedtuple, deque
from neural_net import net
from memory import memory, transition_values
from tools import transition_values

class dqn(nn.Module):
    def __init__(self, discount_rate = 0.9, epsilon = 1.0, learning_rate = 0.001, n_actions = 2,
                    in_features = 1, out_features = 1, hidden_layers_dims = [1], buffer_size = 100000, 
                    batch_size = 64, environment = None, epsilon_min = 0.01, epsilon_decay = 5000, 
                    file_path = None):
        super().__init__()
        self.device = T.device("cuda")
        #use named_tuple
        #be mindful of NaN values
        #use T.save()
        #only works with discrete action space
        #to net device instead of initializing new device

        if environment:
            self.env = environment
            #do something

        self.gamma = discount_rate
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decr = (epsilon - epsilon_min)/epsilon_decay
        self.action_space = [i for i in range(n_actions)]
        
        self.q_eval = net(in_features = in_features, out_features = out_features, hidden_layers_dims = hidden_layers_dims, learning_rate = learning_rate)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = memory(buffer_size)
        self.memory.memory_buffer
        self.transit_vals = transition_values(0.0, 0, 0.0, 0.0, 1.0)

        self.to(self.device)

        #debugging space
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.reward = 0.0
        self.alpha_decay_rate = 0.999

        self.update_interval = 10

        #fix duplicate actions
        #start checking when epsilon < 0.5
        self.duplicate_cnt = 0
        self.dup_buffer_size = 100
        self.duplicate_buffer = T.zeros(self.dup_buffer_size, dtype = T.int32).to(self.device)
        self.last_interval_reward = 0
        self.max_dup_rate = 0.9   #current input_space_size is 18
        self.epsilon_incr_rate = 1.1
        self.alpha_incr_rate = 0.25   #not being used yet
        self.loss = 0.0
        self.lr_scheduler = ls.ExponentialLR(self.q_eval.optimizer, 0.99)
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def transition(self, current_state, action, next_state, reward, done):
        #set to 1.0 when false to do element-wise product
        self.reward += reward
        self.transit_vals = transition_values(current_state, action, next_state, reward, 0.0 if done else 1.0)
        self.memory.update(self.transit_vals)

    def decision(self, obs):
        self.epsilon -= self.epsilon_decr if self.epsilon > self.epsilon_min else 0.0
        if np.random.rand() > self.epsilon:
            #error if use double (from net.forward())
            actions = self.q_eval.forward(T.flatten(T.tensor(np.array(obs)).float()).to(self.q_eval.device))
            action = T.argmax(actions).item()

            ''''''''''''''''''''''''''''''''''''''''''
            #fix epsilon only 
            #if fail, try fixing alpha also
            self.duplicate_buffer[self.duplicate_cnt % self.dup_buffer_size] = action
            self.duplicate_cnt += 1
            
            ''''''''''''''''''''''''''''''''''''''''''

            return action
        return np.random.choice(self.action_space)
        
    def learn(self):
        #the slowest function, try reducing .numpy()
        if self.memory.cur_mem_p < self.batch_size or self.memory.cur_mem_p % self.update_interval != 0:
            return

        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        if self.epsilon < 0.5 and T.mode(self.duplicate_buffer, 0).indices > self.max_dup_rate * self.dup_buffer_size:
            print(self.max_dup_rate * self.dup_buffer_size)
            self.epsilon *= self.epsilon_incr_rate
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        #transitions_batch = transition_values(*zip(*(memory.sample(self.batch_size))))
        #need indices for later
        indices, minibatch = self.memory.sample(self.batch_size)

        #transpose batch to get each individual batch
        #batch_list = transition_values(*zip(*minibatch)).current_state
        current_state_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).current_state, dtype = np.float32)).to(self.q_eval.device)
        action_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).action, dtype = np.float32)).to(self.q_eval.device)
        next_state_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).next_state, dtype = np.float32)).to(self.q_eval.device)
        reward_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).reward, dtype = np.float32)).to(self.q_eval.device)
        terminal_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).done, dtype = np.float32)).to(self.q_eval.device)

        #indices must be long, int, bool
        action_batch = action_batch.long()

        #terminal state is not included -> fixed
        '''
        for i in indices:
            policy = self.q_eval.forward(current_state_batch[i])[action_batch[i]]
            if terminal_batch[i]:
                target[j] = (reward_batch[i])
                continue
            target[j] = (reward_batch[i] + self.gamma * max(self.q_eval.forward(next_state_batch[i])))
            j += 1
        '''
        policy = (self.q_eval.forward(T.flatten(current_state_batch, start_dim = 1))[indices, action_batch]).to(self.device)
        target = reward_batch + (T.mul(T.max(self.q_eval.forward(T.flatten(next_state_batch, start_dim = 1)), 1).values, self.gamma)) * terminal_batch.to(self.q_eval.device)
        #print(target)

        loss = self.q_eval.loss(target, policy).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.loss = loss
        self.lr_scheduler.step()
        self.alpha = self.lr_scheduler.get_last_lr()
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def modify(self, save_before_modifying = True):
        if save_before_modifying:
            a = 1
            #save model

    def save(self, path):
        a = 1

    def load(self, path):
        a = 1
