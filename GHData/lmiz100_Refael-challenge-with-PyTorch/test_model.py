# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:38:11 2019

@author: lior_
"""

import torch
import torch.nn.functional as F
from model import ActorCritic
from torch.autograd import Variable
import time
from collections import deque
from Interceptor_V2 import Init, Game_step, Draw
from train import get_vals, get_vals2

def isnt_vertical(p):
    return p[0] != -2000
    

#test model1(ActorCritic)
def test(rank, params, shared_model, print_num):
    torch.manual_seed(params.seed + rank) # asynchronizing the test agent
    
    model = ActorCritic(params.input_size, 4) # creating one model
    model.eval() # putting the model in "eval" model because it won't be trained
    Init()
    state = torch.zeros(1, params.input_size)
    reward_sum = 0 # initializing the sum of rewards to 0
    done = True # initializing done to True
    start_time = time.time() # getting the starting time to measure the computation time
    actions = deque(maxlen=100) # cf https://pymotw.com/2/collections/deque.html
    episode_length = 0 # initializing the episode length to 0
    last_score = 0
    with torch.no_grad():
        while True: # repeat
            episode_length += 1 # incrementing the episode length by one
            if done: # synchronizing with the shared model (same as train.py)
                model.load_state_dict(shared_model.state_dict())
                cx = Variable(torch.zeros(1, 256)) #128
                hx = Variable(torch.zeros(1, 256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)
            value, action_value, (hx, cx) = model((Variable(state), (hx, cx)))
            prob = F.softmax(action_value, dim=1)
            action = prob.max(1)[1].data
            r_locs, i_locs, c_locs, ang, score = Game_step(action[0])
            state_vals = get_vals(r_locs, i_locs, c_locs, ang, score)
            if episode_length % 300 == 0 and print_num == 9: 
#                Draw()
                print(prob)
                print(any([isnt_vertical(p) for p in i_locs]), ang, score)
#                print(r_locs, i_locs, c_locs, ang, score)
            state = torch.Tensor(state_vals).float().unsqueeze(0)
            done = (episode_length >= 1000)
            reward = score - last_score
            last_score = score
            reward_sum += reward
            if done: # printing the results at the end of each part
                if score > -150:
                    print("Time {}, score {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), score, episode_length))
                reward_sum = 0 # reinitializing the sum of rewards
                episode_length = 0 # reinitializing the episode length
                actions.clear() # reinitializing the actions
                Init()
                state = torch.zeros(1, params.input_size)
                last_score = 0
                return score



# test model2
def test2(ai, print_num):    
    model = ai.brain
    model.eval() # putting the model in "eval" model because it won't be trained
    Init()
    state = torch.zeros(1, ai.params.input_size)
    reward_sum = 0 # initializing the sum of rewards to 0
    done = True # initializing done to True
    start_time = time.time() # getting the starting time to measure the computation time
    actions = deque(maxlen=100) # cf https://pymotw.com/2/collections/deque.html
    episode_length = 0 # initializing the episode length to 0
    last_score = 0
    not_vertical_shot = 0
    
    with torch.no_grad():
        while True: # repeat
            episode_length += 1 # incrementing the episode length by one
            value, action_value = model(Variable(state))
            prob = F.softmax(action_value, dim=1)
            action = prob.max(1)[1].data
            r_locs, i_locs, c_locs, ang, score = Game_step(action[0], train_mode=0)
            state_vals, i_ang = get_vals2(r_locs, i_locs, c_locs, ang, score)
            if i_ang != 90: not_vertical_shot = 1
            if episode_length == 950 and not_vertical_shot == 1 and print_num == 9:
                print("not vertical shot")
#            if episode_length % 300 == 0 and print_num == 9: 
#                Draw()
#                print(prob)
#                print(any([isnt_vertical(p) for p in i_locs]), ang, score)
#                print(r_locs, i_locs, c_locs, ang, score)
            state = torch.Tensor(state_vals).float().unsqueeze(0)
            done = (episode_length >= 1000)
            reward = score - last_score
            last_score = score
            reward_sum += reward
            if done: # printing the results at the end of each part
                if score > -100:
                    print("Time {}, score {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), score, episode_length))
                actions.clear() # reinitializing the actions
                return score
            
            

# test model3
def test3(ai, print_num):    
    model = ai.brain
    model.eval() # putting the model in "eval" model because it won't be trained
    Init()
    state = torch.zeros(1, ai.params.input_size)
    reward_sum = 0 # initializing the sum of rewards to 0
    done = True # initializing done to True
    start_time = time.time() # getting the starting time to measure the computation time
    actions = deque(maxlen=100) # cf https://pymotw.com/2/collections/deque.html
    episode_length = 0 # initializing the episode length to 0
    last_score = 0
    not_vertical_shot = 0
    
    with torch.no_grad():
        while True: # repeat
            episode_length += 1 # incrementing the episode length by one
            if done: # synchronizing with the shared model (same as train.py)
                cx = Variable(torch.zeros(1, 16)) #128
                hx = Variable(torch.zeros(1, 16))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)
            value, action_value, (hx, cx) = model((Variable(state), (hx, cx)))
            prob = F.softmax(action_value, dim=1)
            action = prob.max(1)[1].data
            r_locs, i_locs, c_locs, ang, score = Game_step(action[0])
            state_vals, i_ang = get_vals2(r_locs, i_locs, c_locs, ang, score)
            if i_ang != 90: not_vertical_shot = 1
            if episode_length == 950 and not_vertical_shot == 1 and print_num == 9:
                print("agent3 not vertical shot")
#            if episode_length % 300 == 0 and print_num == 9: 
#                Draw()
#                print(prob)
#                print(any([isnt_vertical(p) for p in i_locs]), ang, score)
#                print(r_locs, i_locs, c_locs, ang, score)
            state = torch.Tensor(state_vals).float().unsqueeze(0)
            done = (episode_length >= 1000)
            reward = score - last_score
            last_score = score
            reward_sum += reward
            if done: # printing the results at the end of each part
                if score > -100:
                    print("agent3 Time {}, score {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), score, episode_length))
                actions.clear() # reinitializing the actions
                return score