# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:23:20 2019

@author: lior_
"""

# Training the AI

import torch
import torch.nn.functional as F
from model import ActorCritic
from torch.autograd import Variable
from Interceptor_V2 import Init, Game_step, Draw
import math



#global variables
city_locations = []
t_x = -2000 # turret x val
t_y = 0     # turret y val
hostile_x, hostile_y = [4800, 0]    #hostile location
angle = 0

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

     
def takeFirst(elem):
    
    return elem[0]


def cityDist(elem):
    min_city_dist = math.sqrt(((hostile_x - t_x)**2) + ((hostile_y - t_y)**2))
    
    for city in city_locations:
        c_x, c_y = city
        
        cur_dist = math.sqrt(((elem[0] - c_x)**2) + ((elem[1] - c_y)**2))
        min_city_dist = min(min_city_dist, cur_dist)
        
    return min_city_dist


def angleDiff(elem):
    r_ang = math.atan((elem[0] - t_x) / (elem[1] + 0.00001)) * 57.295
    ang_diff = r_ang - angle
    
    return math.fabs(ang_diff)
    

def convertDistance(d):
    return round(d / 1000)


def convertAngle(a):
    return (a // 6)



# return inputs for model1(ActorCritic)
def get_vals(r_locs, i_locs, c_locs, ang, score):
    global city_locations, angle
    city_locations = c_locs
    angle = ang
    
    res = [0 for x in range (0,65)]
    
    ind = 0
    sorted_rockets = sorted(sorted(r_locs, key = cityDist)[0 : 22], key = angleDiff)[0 : 16] #sort by distance from city and than by consider angle
    # consider 8 first sorted rockets
    for r in sorted_rockets:
        x, y = r
        
        dist = math.sqrt(((x - t_x)**2) + ((y - t_y)**2))
        r_ang = math.atan((x - t_x) / (y + 0.00001)) * 57.295   # multiple to convert from radians to degree
        ang_diff = r_ang - ang
        
        res[ind] = convertDistance(dist)
        res[ind + 1] = convertAngle(r_ang)
#        res[ind + 2] = ang_diff
        ind += 2
    
       
    dist = math.sqrt(((hostile_x - t_x)**2) + ((hostile_y - t_y)**2))
    r_ang = math.atan((hostile_x - t_x) / (hostile_y + 0.00001)) * 57.295   # multiple to convert from radians to degree
    ang_diff = r_ang - ang
        
    #while ind < len(res) - 2:
    for _ in range(16 - len(sorted_rockets)):
        res[ind] = convertDistance(dist)
        res[ind + 1] = convertAngle(r_ang)
#        res[ind + 2] = ang_diff
        ind += 2
      
    for interceptor in i_locs[0 : 16]: 
        dist = math.sqrt(((interceptor[0] - t_x)**2) + ((interceptor[1] - t_y)**2))
        i_ang = math.atan((interceptor[0] - t_x) / (interceptor[1] + 0.00001)) * 57.295
        
        res[ind] = convertDistance(dist)
        res[ind + 1] = convertAngle(i_ang)
        ind += 2
        
        
    i_ang = ang
    dist = 0
    
    for _ in range(16 - len(i_locs[0 : 16])):
        res[ind] = 0
        res[ind + 1] = convertAngle(i_ang)
        ind += 2
    
        
    res[-1] = convertAngle(ang)
    
    return res


# return inputs for model2, model3
def get_vals2(r_locs, i_locs, c_locs, ang, score):
    if len(r_locs) == 0:
        rocket = [hostile_x, hostile_y]  #hostile location
    else: 
        rocket = r_locs[0]
        
    dist_r = math.sqrt(((rocket[0] - t_x)**2) + ((rocket[1] - t_y)**2))
    r_ang = math.atan((rocket[0] - t_x) / (rocket[1] + 0.00001)) * 57.295   # multiple to convert from radians to degree
    ang_diff = r_ang - ang
    
    i_ang = -1
    if len(i_locs) == 0:
        dist_interceptor2rocket = -1
    else:
        dist_interceptor2rocket = math.sqrt(((i_locs[0][0] - rocket[0])**2) + ((i_locs[0][1] - rocket[1])**2))
        i_ang = math.atan((i_locs[0][0] - t_x) / (i_locs[0][1] + 0.00001)) * 57.295   # multiple to convert from radians to degree
      
     # 0 <= angle <=180 rather than -90 <= angle <= 90
    return [r_ang + 90, dist_r, ang_diff, dist_interceptor2rocket, ang + 90], (i_ang + 90)    
        


def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank)    
    model = ActorCritic(params.input_size, 4)
    Init()
    state = torch.zeros(1, params.input_size)
    done = True
    episode_length = 0
    last_score = 0
    completed_episodes = 0
    same_angle = 0
    last_angle = 0
    
    while completed_episodes < 20:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256)) #128
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(params.num_steps):
            value, action_values, (hx, cx) = model((Variable(state), (hx, cx)))
            prob = F.softmax(action_values, dim=1)            
            log_prob = F.log_softmax(action_values, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial(1)
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_probs.append(log_prob)
            r_locs, i_locs, c_locs, ang, score = Game_step(action[0,0])
            state_vals = get_vals(r_locs, i_locs, c_locs, ang, score)
            state = torch.Tensor(state_vals).float().unsqueeze(0)
            done = (episode_length >= params.max_episode_length)
            reward = score - last_score
#            if action[0,0] == 3: reward += 20
#            if action[0,0] == 3 and (-15<ang<0 or 0<ang<15): reward += 10
            if score > 0: reward += 5
#            same_angle = same_angle + 1 if last_angle == ang and same_angle < 7 else 0
#            last_angle = ang
#            reward = score - last_score if same_angle < 6 else score - last_score - 7
            #reward = max(min(reward, 1), -1) #keep reward in range [-1, 1]
#            if action[0,0] == 3: print("fire in the hole!")
#            reward = (score + 0.6 - last_score) if (action[0,0] == 3 and ang != 0) else (score - last_score)
            rewards.append(reward)
            last_score = score
            #reward = max(min(reward, 1), -1)
            if done:
                episode_length = 0
                Init()
                state = torch.zeros(1, params.input_size)
                last_score = 0
                completed_episodes += 1
#                print(completed_episodes)
                break
        
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state), (hx, cx)))
            R = value.data
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]
        optimizer.zero_grad()
#        (0.67 * policy_loss + 0.33 * value_loss).backward()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()
 
    
       
#train model2
def train2(ai):
    model = ai.brain
    model.train()
    Init()
    state = torch.zeros(1, ai.params.input_size)
    done = True
    episode_length = 0
    last_score = 0
    completed_episodes = 0
    last_interceptor_dist = -1
    while completed_episodes < 20:
        episode_length += 1
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(ai.params.num_steps):
            value, action_values = model(Variable(state))
            prob = F.softmax(action_values, dim=1)            
            log_prob = F.log_softmax(action_values, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial(1)
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_probs.append(log_prob)
            r_locs, i_locs, c_locs, ang, score = Game_step(action[0,0], train_mode=1)
            state_vals, i_ang = get_vals2(r_locs, i_locs, c_locs, ang, score)
            state = torch.Tensor(state_vals).float().unsqueeze(0)
            done = (episode_length >= ai.params.max_episode_length)
            reward = score - last_score
            if score > 0: reward += 1
            if i_ang > 90:
                if (last_interceptor_dist > state_vals[3] > -1) or (last_interceptor_dist == -1 and state_vals[3] > -1): reward += 0.5
            if state_vals[3] > last_interceptor_dist > -1: reward -= 0.15
            if 10 < state_vals[2] < 30 : 
                reward += 0.1
            elif state_vals[2] < 0: reward -= 0.6
            last_interceptor_dist = state_vals[3]
            rewards.append(reward)
            last_score = score
#            print_r = r_locs[0] if len(r_locs) > 0 else [-1, -1]
#            if episode_length % 6 == 0:
#                print("enemy ang {}, turret ang {}, num of interceptors is {}, score {}, enemy place {}".format(state_vals[0], state_vals[4], len(i_locs), score, print_r))
#                Draw()
            if done:
                episode_length = 0
                Init()
                state = torch.zeros(1, ai.params.input_size)
                last_score = 0
                completed_episodes += 1
                last_interceptor_dist = -1
                break
        
        R = torch.zeros(1, 1)
        if not done:
            value, _ = model(Variable(state))
            R = value.data
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = ai.params.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            TD = rewards[i] + ai.params.gamma * values[i + 1].data - values[i].data
            gae = gae * ai.params.gamma * ai.params.tau + TD
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]
        ai.optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
        ai.optimizer.step()
        
        
        
#train model3
def train3(ai):
    model = ai.brain
    model.train()
    Init()
    state = torch.zeros(1, ai.params.input_size)
    done = True
    episode_length = 0
    last_score = 0
    completed_episodes = 0
    last_interceptor_dist = -1
    while completed_episodes < 20:
        episode_length += 1
        if done:
            cx = Variable(torch.zeros(1, 16)) 
            hx = Variable(torch.zeros(1, 16))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(ai.params.num_steps):
            value, action_values, (hx, cx) = model((Variable(state), (hx, cx)))
            prob = F.softmax(action_values, dim=1)            
            log_prob = F.log_softmax(action_values, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial(1)
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_probs.append(log_prob)
            r_locs, i_locs, c_locs, ang, score = Game_step(action[0,0], train_mode=1)
            state_vals, i_ang = get_vals2(r_locs, i_locs, c_locs, ang, score)
            state = torch.Tensor(state_vals).float().unsqueeze(0)
            done = (episode_length >= ai.params.max_episode_length)
            reward = score - last_score
            if score > 0: reward += 1
            if i_ang > 90:
                if (last_interceptor_dist > state_vals[3] > -1) or (last_interceptor_dist == -1 and state_vals[3] > -1): reward += 0.5
            if state_vals[3] > last_interceptor_dist > -1: reward -= 0.15
            if 10 < state_vals[2] < 30 : 
                reward += 0.2
            elif state_vals[2] < 0: reward -= 0.6
            last_interceptor_dist = state_vals[3]
            rewards.append(reward)
            last_score = score
            if done:
                episode_length = 0
                Init()
                state = torch.zeros(1, ai.params.input_size)
                last_score = 0
                completed_episodes += 1
                last_interceptor_dist = -1
#                print(completed_episodes)
                break
        
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state), (hx, cx)))
            R = value.data
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        
        for i in reversed(range(len(rewards))):
            R = ai.params.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            TD = rewards[i] + ai.params.gamma * values[i + 1].data - values[i].data
            gae = gae * ai.params.gamma * ai.params.tau + TD
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]
            
        ai.optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
        ai.optimizer.step()