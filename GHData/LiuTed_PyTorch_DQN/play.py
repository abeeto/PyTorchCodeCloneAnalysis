import DQN
import Param
import EpsScheduler
import gym
from gym import logger, wrappers
import torch
import numpy as np
import time

if __name__ == '__main__':
    env = Param.get_env()
    logger.set_level(logger.INFO)
    env = wrappers.Monitor(env, directory='/tmp/DQN', force=True)
    eps_sche = EpsScheduler.EpsScheduler(0.1, 'Fixed')
    dqn = DQN.DQN(Param.MEMORY_SIZE, env, eps_sche)
    dqn.restore('net.pkl')

    avgstep = 0    
    for i in range(Param.TEST_EPISODE):
        state = env.reset()
        if dqn.state_based:
            state[0] /= 4.8
            state[1] /= 4.8
            state[2] /= 0.418 
            state[3] /= 0.418 # location and speed normalization
        else:
            screen = Param.get_screen(env)
            last_screen = Param.get_screen(env)
            state = torch.cat([last_screen, screen], 0)
        done = False
        step = 0
        while not done:
            if dqn.state_based:
                action = dqn.get_action(torch.tensor([state], dtype=torch.float32).to(Param.device), False)
                next_state, reward, done, _ = env.step(action.item())
                next_state[0] /= 4.8
                next_state[1] /= 4.8
                next_state[2] /= 0.418 
                next_state[3] /= 0.418
            else:
                action = dqn.get_action(torch.unsqueeze(state, 0), False)
                _, reward, done, _ = env.step(action.item())
                last_screen = screen
                screen = Param.get_screen(env)
                next_state = torch.cat([last_screen, screen], 0)
            state = next_state
            step += 1
        avgstep += step
        print(i, step)
    avgstep /= Param.TEST_EPISODE
    print('Average:', avgstep)
    env.close()
