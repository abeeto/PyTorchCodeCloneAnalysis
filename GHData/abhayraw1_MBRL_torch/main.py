import pdb
import gym
import go2goal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import models2 as Models
import numpy as np

np.set_printoptions(suppress=True, linewidth=300, precision=4,
                    formatter={'float_kind':'{:10.6f}'.format})


def get_states(env):
    goal, pose = env.goal.tolist(), env.agent.pose.tolist()
    dess = [*goal[:2], np.cos(goal[2]), np.sin(goal[2]), 0, 0]
    curr = [*pose[:2], np.cos(pose[2]), np.sin(pose[2]), *env.agent.vel]
    return torch.tensor(dess).double(), torch.tensor(curr).double()


def get_obs_tensor(current_pose, desired_pose, v, w):
    a, b = current_pose.getPoseInFrame(desired_pose)
    return torch.tensor([*a, *b, v, w]).double()


def optimize(loss, policy, optimizer):
    policy.train()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 15.)
    optimizer.step()


def compute_loss(actions, states, target): 
    gloss, aloss, vloss, nloss = 0, 0, 0, 0
    tloss = 0
    seq_len = len(actions)
    for i, (s, u) in enumerate(zip(states, actions)):
        # print(i)
        gloss += ((i+1)/seq_len)**2*torch.norm((s - target)[:2])# + 0.3*torch.norm(u)
        # nloss += (s[-2] < 0)*torch.norm(s[-2])*(i+1)/seq_len
        aloss += torch.norm(u)*0.05
        vloss += torch.norm(s[-2:])*0.05
        if i == (seq_len - 10):
            tloss += torch.norm((states[-1][-2:] - target[-2:]))

    # if torch.norm(states[-1][:2] - target[:2]) < 0.1:
        # gloss = gloss/2
        # gloss += tloss
    # else:

    loss = gloss*2 + aloss + vloss + tloss
    print(f'Goal Loss: {gloss}')
    print(f'Control Loss: {aloss}')
    # print(f'Terminal Loss: {tloss}')
    print(f'Velocity Loss: {vloss}')
    # print(f'Negative Velocity Loss: {nloss}')
    print(f'Total Loss: {loss}')
    print('='*100)
    return loss #+ nloss# + tloss


def do_actions(actions, env, render=False, states=None):
    low, high = [-0.1, -np.pi/4], [0.3, np.pi/4]
    for ac in actions:
        # if render:
        #     pdb.set_trace()
        env.step(env.agent.vel)
        env.agent.vel += ac*env.dt
        env.agent.vel = np.clip(env.agent.vel, low, high)
        if render:
            env.render()
            # time.sleep(0.1)


def main(env, policy, optimizer, n_eps, evaluate=False):
    for eps in range(n_eps):
        env.reset()
        # env.agent.pose.theta = env.agent.pose.theta = np.arctan2(*env.obs['desired_goal'][:2][::-1]) + np.pi/2
        env_goal = env.goal.tolist()
        MAX_HORIZON = 50
        # prev_pose = env.agent.pose.clone()
        # curr_pose = env.agent.pose.clone()
        # dsrd_pose = env.goal
        prev_state = None
        current_state = None
        desired_state = None
        for t in range(env._max_episode_steps):
            for _ in range(10):
                desired_state, current_state = get_states(env)
                v, w = env.agent.vel
                actions = []
                traj = []
                # actions, states = policy(get_obs_tensor())
                actions, states = policy(current_state, prev_state, desired_state)
                loss = compute_loss(actions, states, desired_state)
                if evaluate:
                    break
                else:
                    optimize(loss, policy, optimizer)
                traj = np.array([s.detach().numpy() for s in states])[:, :2]
                env.render(extra=traj)
            actions = np.array([a.detach().numpy() for a in actions])
            if t == (env._max_episode_steps - 1):
                # pdb.set_trace()
                do_actions(actions, env, render=True, states=np.array([s.detach().numpy() for s in states]))
                break
            do_actions(actions[:5], env, render=True)
            # prev_state = current_state.clone()
            if np.linalg.norm(env.distance_from_goal()) < 0.05:
                break
        if (eps + 1)%50 == 0:
            evaluate(env, policy, 10)
            torch.save(
                policy.state_dict(),
                f'models/pi{time.strftime("%d%m%Y_%H%M", time.localtime())}'
            )


if __name__ == '__main__':
    EVALUATE = False
    N_EPS = 2500
    env = gym.make('Go2Goal-v0', config={'num_iter': 1, 'dt': 0.1})
    policy = Models.MBRLPolicy2(50, 0.1)
    optimizer = torch.optim.Adam(policy.policy.parameters(), lr=0.0025)
    policy.load_state_dict(torch.load('/home/aarg/Documents/mbrl_torch_g2g/models/pi30082019_2303'))
    # evaluate(env, policy, 10)
    main(env, policy, optimizer, N_EPS, evaluate=EVALUATE)
    env.close()
    torch.save(policy.state_dict(), f'pi{time.strftime("%d%m%Y_%H%M", time.localtime())}')