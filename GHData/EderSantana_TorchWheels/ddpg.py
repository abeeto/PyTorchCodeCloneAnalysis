import numpy as np
import random
import argparse
import json
import timeit

import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from gym_torcs import TorcsEnv

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork, train_actor_target
from CriticNetwork import CriticNetwork, train_critic_target
from OU import OU

OU = OU()  # Ornstein-Uhlenbeck Process
cuda = True

import ipdb

def playGame(train_indicator=1):  # 1: Train, 0: Test
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target network hyperparam
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Learning rate for Critic

    action_dim = 3  # steering, accel, brake
    state_dim = 29  # number of sensors

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    actor = ActorNetwork(state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    actor_target = ActorNetwork(state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    critic_target = CriticNetwork(state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    actor.cuda()
    actor_target.cuda()
    critic.cuda()
    critic_target.cuda()
    buff = ReplayBuffer(BUFFER_SIZE)

    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    actor_opt = optim.Adam(actor.parameters(), lr=actor.LEARNING_RATE)
    critic_opt = optim.Adam(critic.parameters(), lr=critic.LEARNING_RATE)

    print('Loading network weights')
    try:
        actor.load_state_dict(torch.load('actormodel.h5'))
        critic.load_state_dict(torch.load('criticmodel.h5'))
        actor_target.load_state_dict(torch.load('actortarget.h5'))
        critic_target.load_state_dict(torch.load('critictarget.h5'))
        print('Network weights loaded!')
    except:
        print('Failed to load weights!')

    print('TORCS Experiment Start.')
    for i in range(episode_count):
        print("Episode : {}, Replay Buffer : {}".format(i, buff.count()))
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100., ob.rpm))
        # s_t = np.zeros(state_dim)

        total_reward = 0.
        for j in range(max_steps):
            s_t = s_t.astype(np.float32)
            s_t_c = torch.from_numpy(s_t[None, :])
            s_t_c = s_t_c.cuda()
            s_t_c = Variable(s_t_c)

            loss = 0
            epsilon -= 1.0 / EXPLORE
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor(s_t_c)
            a_t_original = a_t_original.cpu().data.numpy()
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0, 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5, 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

            a_t = a_t_original + noise_t

            ob, r_t, done, info = env.step(a_t[0])
            # r_t = 0.
            # done = 0.
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100., ob.rpm))
            # s_t1 = np.zeros(state_dim)
            s_t1 = s_t1.astype('float32')
            buff.add(s_t, a_t[0], r_t, s_t1, done)

            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            new_states = Variable(torch.from_numpy(new_states.astype('float32')).cuda())
            target_a_values = actor_target(new_states)
            target_q_values = critic_target(new_states, target_a_values)

            target_q_values = target_q_values.cpu().data.numpy()
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            y_t = Variable(torch.from_numpy(y_t.astype('float32')).cuda())

            if train_indicator:
            # if False:
                states_c = Variable(torch.from_numpy(states.astype('float32')).cuda())
                actions_c = Variable(torch.from_numpy(actions.astype('float32')).cuda())

                # Train critic
                critic_opt.zero_grad()
                out = critic(states_c, actions_c)
                LOSS = F.mse_loss(out, y_t)
                LOSS.backward()
                critic_opt.step()

                actor_opt.zero_grad()
                out = critic(states_c, actions_c)
                REWARD = -torch.mean(out)
                REWARD.backward()
                loss += LOSS.data[0] 
                train_critic_target(critic, critic_target)

                # Train actor
                actor_opt.step()
                train_actor_target(actor, actor_target)


            total_reward += r_t
            s_t = s_t1

            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if train_indicator:
                print("Saving the models.")
                torch.save(actor.state_dict(), 'actormodel.h5')
                torch.save(critic.state_dict(), 'criticmodel.h5')
                torch.save(actor_target.state_dict(), 'actortarget.h5')
                torch.save(critic_target.state_dict(), 'critictarget.h5')

        print("Total Reward @ {}-th episode: {}".format(i, total_reward))
        print("Total Step: {}".format(step))
        print("")

    env.end()
    print("Finish")


if __name__ == "__main__":
    playGame()
