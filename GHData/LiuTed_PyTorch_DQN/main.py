import DQN
import Param
import EpsScheduler
import gym
from gym import logger, wrappers
import torch
import numpy as np
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    env = Param.get_env()
    logger.set_level(logger.INFO)
    env = wrappers.Monitor(env, directory='/tmp/DQN', force=True)
    writer = SummaryWriter('./Result')
    eps_sche = EpsScheduler.EpsScheduler(1., 'Linear', lower_bound=0.1, target_steps=100000)
    dqn = DQN.DQN(Param.MEMORY_SIZE, env, eps_sche)
    avgloss = 0.
    cnt = 0
    i = 0
    loop = 0
    episode_step = 0
    eps_to_win = 100
    while i < Param.NUM_EPISODE:
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
                action = dqn.get_action(torch.tensor([state], dtype=torch.float32).to(Param.device))
                next_state, reward, done, _ = env.step(action.item())
            else:
                action = dqn.get_action(torch.unsqueeze(state, 0))
                _, reward, done, _ = env.step(action.item())
            step += 1

            if done:
                next_state = None
            else:
                if dqn.state_based:
                    next_state[0] /= 4.8
                    next_state[1] /= 4.8
                    next_state[2] /= 0.418 
                    next_state[3] /= 0.418
                else:
                    last_screen = screen
                    screen = Param.get_screen(env)
                    next_state = torch.cat([last_screen, screen], 0)
            
            dqn.push([
                state,
                action,
                next_state,
                reward
            ])
            state = next_state

            loss = dqn.learn()
            if loss is not None:
                writer.add_scalar('Train/loss', loss.item(), dqn.step)
        writer.add_scalar('Train/length', step, dqn.step)
        writer.add_scalar('Train/eps', dqn.eps_scheduler.eps, dqn.step)
        for idx, param in enumerate(dqn.policy.parameters()):
            writer.add_histogram('Train/Param%d'%idx, param.data, dqn.step)

        episode_step += step
        if(episode_step >= Param.STEP_PER_EPISODE):
            episode_step = 0
            i += 1

        if loop % Param.DO_TEST_EVERY_LOOP == 0:
            avgstep = 0    
            for _ in range(Param.TEST_EPISODE):
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
            avgstep /= Param.TEST_EPISODE
            writer.add_scalar('Test/length', avgstep, dqn.step)
            if avgstep >= 195.0:
                print('Solved!')
                dqn.save('net.pkl')
                break
        loop += 1
        
    env.close()



