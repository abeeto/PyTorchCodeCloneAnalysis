# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:16:02 2019

@author: lior_
"""

from __future__ import print_function
import os
import torch
from model import ActorCritic, Ai
from train import train, train2, train3
from test_model import test, test2, test3
import my_optimizer
import time
import logging
import numpy as np
import math
from model2 import Ai2
from model3 import Ai3
#import torch.multiprocessing as mp
#from multiprocessing.pool import ThreadPool
from Interceptor_V2 import Init, Draw, Game_step
from train import get_vals
#import threading

'''
from torch.autograd import Variable
import time
from collections import deque
from Interceptor_V2 import Init, Draw, Game_step
from train import get_vals
'''


# Gathering all the parameters (that we can modify to explore)
class Params():
    def __init__(self, lr, steps, input_size):
        #reward between -1 to 4
        self.lr = lr    #originally 0.0001 ult 0.7, new NN 0.8
        self.gamma = 0.99  # originally 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 16
        self.num_steps = steps
        self.max_episode_length = 1100 // self.num_steps
        self.input_size = input_size
        self.output_size = 4
        

# save model as pth file for future reuse
def save(model, optimizer, lr, steps, training_num):
    training = 0
    if os.path.isfile('last_brain{}s{}.pth'.format(lr,steps)):
        training = torch.load('last_brain{}s{}.pth'.format(lr,steps))['training_completed']
        
    torch.save({'lr': lr,
                'steps': steps,
                'training_completed': training + training_num,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
                   }, 'last_brain{}s{}.pth'.format(lr,steps))
    
 
# load saved model with learning rate (lr) and steps parameters
def load(model, optimizer, lr, steps):
    if os.path.isfile('last_brain{}s{}.pth'.format(lr,steps)):
        print("=> loading checkpoint... ")
        checkpoint = torch.load('last_brain{}s{}.pth'.format(lr,steps))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("load done!")
        return checkpoint['state_dict'], checkpoint['optimizer']
    else:
        print("no checkpoint found...")
        return model.state_dict(), optimizer.state_dict()

        
       
    
logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")  

# automated train and test for model2, model3    
#optional_lr = [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#optional_steps = [12, 16, 20]
#model2 good results in (lr, steps): (0.001, 16), (0.0001, 20), (0.8, 16)
lr = [0.0001, 0.001, 0.1, 0.7] #[ 0.2, 0.3, 0.4] [0.0001, 0.001, 0.8]
steps = [16, 20] #[12, 16, 20]
test_with = [(0.0001, 20), (0.001, 16), (0.8, 16), (0.8, 20)] #(0.8, 16)


#for (l, s) in test_with: # when want test of specific (lr, steps) from test_with list
for l in lr:
    for s in steps:
        params = Params(l, s, 5)
        agent2 = Ai2(params)
        agent3 = Ai3(params)
        
        stop_ag2 = 0
        stop_ag3 = 0
        avg2 = -1
        avg3 = -1
    
        training_num = 26
        for ind in range (1, training_num):
            if ind == 1:
                logging.info("########  testing lr ={}, steps ={}  ########\n".format(l, s))
                 
            
            if stop_ag2 == 0:
                train2(agent2)
                avg2 = np.mean([test2(agent2, x) for x in range(10)])
                agent2.update_game_score(avg2)
            if  stop_ag3 == 0:
                train3(agent3) 
                avg3 = np.mean([test3(agent3, x) for x in range(10)])
                agent3.update_game_score(avg3)
            
            logging.info("iteration {}, avg2: {}, avg3: {} \n".format(ind, avg2, avg3))
    #            if (ind == 11 and avg < -350) or (ind > 7 and avg < -1400) :
            if ind > 11 and avg2 < -1400: stop_ag2 = 1
            if ind > 11 and avg3 < -1400: stop_ag3 = 1
            if stop_ag2 == 1 and stop_ag3 == 1:
                logging.info("finished with bad results\n")
                break
            
    #            if ind > 11 and avg3 <= avg2 < -1400:
    #                logging.info("finished with bad results: {}\n".format(avg3))
    #                break
        
        
        time.sleep(90)




    
logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")  
      
#optional_lr = [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#checked lr = [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#good results in (lr, steps): (0.0001, 16),(0.1, 12), (0.1, 16), (0.1, 20), (0.2, 12), (0.3, 20), (0.6, 20), (0.8, 12), (0.9, 20)
lr = [0.0001] #[ 0.2, 0.3, 0.4]
steps = [ 16, 20] #[12, 16, 20]

#automated train and test for model1(ActorCritic)
for l in lr:
    for s in steps:
        os.environ['OMP_NUM_THREADS'] = '1'
        params = Params(l, s, 65)
        torch.manual_seed(params.seed)
        shared_model = ActorCritic(params.input_size, 4)
        #shared_model.share_memory()
        optimizer = my_optimizer.SharedAdam(shared_model.parameters(), lr=params.lr)
        #optimizer.share_memory()

# code for loading saved model        
#        loaded_model, loaded_optimizer = load(shared_model, optimizer, l, s)
#        shared_model.load_state_dict(loaded_model)
#        optimizer.load_state_dict(loaded_optimizer)

        training_num = 26
        for ind in range (1, training_num):
            if ind == 1:
                logging.info("########  testing lr ={}, steps ={}  ########\n".format(l, s))
                
            train(1, params, shared_model, optimizer)
           
            avg = np.mean([test(params.num_processes, params, shared_model, x) for x in range(10)])
            
            logging.info("iteration {}, avg: {}\n".format(ind, avg))
            if (ind == 11 and avg < -350) or (ind > 7 and avg < -1400) :
                logging.info("finished with bad results: {}\n".format(avg))
                break
               
#        save(shared_model, optimizer, l, s, training_num - 1)
        time.sleep(90)





# =============================================================================
# #game test with agent
# params = Params(0.0001, 16)
# torch.manual_seed(params.seed)
# shared_model = ActorCritic(params.input_size, 4)
# optimizer = my_optimizer.SharedAdam(shared_model.parameters(), lr=params.lr)
#  
# loaded_model, loaded_optimizer = load(shared_model, optimizer, params.lr, params.num_steps)
# shared_model.load_state_dict(loaded_model)
# optimizer.load_state_dict(loaded_optimizer)
# agent = Ai(shared_model, optimizer, params)
# 
# 
# score_list = []
# for _ in range(3):
#     score = 0
#     Init()
#     r_locs, i_locs, c_locs, ang, score = Game_step(1)
#     for stp in range(1000):
#         #action_button = *** Insert your logic here: 0,1,2 or 3 ***
#         action_button = agent.select_softmax_action(get_vals(r_locs, i_locs, c_locs, ang, score))
#         if stp < 10: Draw()
# #        if stp < 12:
# #            action_button = 2
# #        else:
# #            action_button = 3
#         r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
#         #Draw()
#     #    if stp == 600:
#         if stp % 200 == 0:
#     #        print(i_locs)
#     #        print(get_vals(r_locs, i_locs, c_locs, ang, score))
#     #        print(len(get_vals(r_locs, i_locs, c_locs, ang, score)))
#             print(ang)
#             #print(r_locs)
#             #print(sorted(r_locs, key = takeFirst))
#             #print(r_locs, i_locs, c_locs, ang, score)
# #            Draw()
#             
#     print("your final score is: {}".format(score))
#     score_list.append(score)
#     agent.update_game_score(score)
#     
# print("your average score is: {}".format(np.mean(score_list)))
# print(agent.get_avg_score())
# =============================================================================


# =============================================================================
# #return brain index if exists or -next_free_index else
# def search_brain(lr, steps):
#     found = 0
#     ind = 1
#     
#     while found == 0:
#         if os.path.isfile('last_brain{}.pth'.format(ind)):
#             checkpoint = torch.load('last_brain{}.pth'.format(ind))
#             if checkpoint['lr'] == lr and checkpoint['steps'] == steps:
#                 found = 1
#             else:   #file exist with differn parameters, keep searching
#                 ind += 1
#         else:   #file doesn't exist and can create one
#             break
#         
#     return ind if found == 1 else -ind
# =============================================================================


# =============================================================================
# for x in range(-3):
#     print("check if print {}".format(x))
# 
# print(np.mean([4,7,19]))
# 
# def my_sleep():
#     time.sleep(10)
#     
#     return
# 
# 
# logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
# 
# #x = threading.Thread(target=my_sleep, args=())
# #logging.info("Main  before running thread")
# #x.start()
# #logging.info("Main : wait for the thread to finish")
# #x.join()
# #logging.info("Main: all done")
# 
# p = mp.Process(target = my_sleep, args = ())
# logging.info("Main : before running proc")
# p.start()
# logging.info("Main : wait for the proc to finish")
# p.join()
# logging.info("Main: all done")
# 
# 
# #my_sleep()
# 
# print("p terminated")
# =============================================================================
    

# =============================================================================
# processes = []
# p = mp.Process(target=test, args=(params.num_processes, params, shared_model))
# p.start()
# processes.append(p)
# for rank in range(0, params.num_processes):
#     p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
#     p.start()
#     processes.append(p)
# for p in processes:
#     p.join()
# 
# 
# print("finish!!!!!!!!!!!!!!!!!!!!!!!!!!")
# 
# for p in processes:
#     print(p.is_alive())
# =============================================================================
  