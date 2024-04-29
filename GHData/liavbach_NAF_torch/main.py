import gym
import torch
import numpy as np
import sys
import os

from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from replay_buffer import ReplayBuffer, Transition

from plot import plot_results

EPISODE_TO_SCORE = 1

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device_name}')
DEVICE = torch.device(device_name)
DTYPE = torch.float

args_mc = {'env_name': 'MountainCarContinuous-v0',
           'seed': 42,
           'gamma': 1,
           'tau': 0.001,
           'hidden_size': 200,
           'replay_size': 1000000,
           'num_episodes': 1000,
           'batch_size': 128,
           'replay_num_updates': 5,
           'ou_noise': True,
           'noise_scale': 3,
           'final_noise_scale': 0,
           'exploration_end': 500,
           'evaluate_episodes': 100}

args_pd = {'env_name': 'Pendulum-v0',
           'seed': 42,
           'gamma': 1,
           'tau': 0.001,
           'hidden_size': 200,
           'replay_size': 20000,
           'num_episodes': 1000,
           'batch_size': 128,
           'replay_num_updates': 5,
           'ou_noise': True,
           'noise_scale': 1,
           'final_noise_scale': 0.1,
           'exploration_end': 400,
           'evaluate_episodes': 100}

args_ll = {'env_name': 'LunarLanderContinuous-v2',
           'seed': 42,
           'gamma': 1,
           'tau': 0.001,
           'hidden_size': 200,
           'replay_size': 100000,
           'num_episodes': 1000,
           'batch_size': 128,
           'replay_num_updates': 5,
           'ou_noise': True,
           'noise_scale': 3,
           'final_noise_scale': 0.1,
           'exploration_end': 500,
           'evaluate_episodes': 100}


def run():
    num_steps = 0

    for episode in range(args['num_episodes']):
        state = env.reset()

        update_noise(episode)

        episode_steps = 0
        is_done = False

        while not is_done:
            act = agent.select_action(state, ounoise)
            suc_state, reward, is_done, _ = env.step(act)

            num_steps += 1
            episode_steps += 1

            done_mask = 0.0 if is_done else 1.0
            replay_buffer.push([state], [act], [done_mask], [suc_state], [reward])

            state = suc_state

            if len(replay_buffer) > args['batch_size']:
                train_on_minibatches()

        if episode % EPISODE_TO_SCORE == 0:
            eval_score = evaluate_policy()

            report_results(episode + 1, num_steps, eval_score)
            print(
                f'Episode: {episode + 1}, Total numsteps: {num_steps}, '
                f'Score: {eval_score}')
        if episode % 5 == 0:
            agent.save_model(args['env_name'])

    env.close()


def run_simulation():
    done = False
    t = 0
    gt = 0

    state = env.reset()

    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        gt += reward * (args['gamma'] ** t)

        if done:
            break

        t += 1

    return gt


def evaluate_policy():
    acc = 0

    for i in range(args['evaluate_episodes']):
        res = run_simulation()
        acc += res

    return acc / args['evaluate_episodes']


def update_noise(episode):
    if ounoise:
        ounoise.scale = (args['noise_scale'] - args['final_noise_scale']) * \
                        max(0, args['exploration_end'] - episode) / args['exploration_end'] + \
                        args['final_noise_scale']
        ounoise.reset()


def train_on_minibatches():
    for i in range(args['replay_num_updates']):
        transitions = replay_buffer.sample(args['batch_size'])
        batch = Transition(*zip(*transitions))
        agent.update_parameters(batch)


def report_results(episode, numsteps, score):
    results_path = 'results'
    os.makedirs(results_path, exist_ok=True)

    file_name = f'results_{args["env_name"]}.csv'
    file_path = os.path.join(results_path, file_name)

    add_head = file_name not in os.listdir(results_path)
    file1 = open(file_path, "a+")
    if add_head:
        file1.write("Episode,TotalSteps,Score\n")

    file1.write(f'{episode},{numsteps},{score}\n')
    file1.close()


if __name__ == '__main__':
    env = sys.argv[1]
    args = None

    if env == 'mc':
        args = args_mc
    elif env == 'pd':
        args = args_pd
    elif env == 'll':
        args = args_ll
    else:
        print('Environment not selected, Please choose from: mc, pd,ll')
        exit(-1)

    env = NormalizedActions(gym.make(args['env_name']))

    env.seed(args['seed'])
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    agent = NAF(args['gamma'], args['tau'], args['hidden_size'],
                env.observation_space.shape[0], env.action_space)
    agent.load_model(f'models/naf_{args["env_name"]}')

    replay_buffer = ReplayBuffer(args['replay_size'])

    ounoise = OUNoise(env.action_space.shape[0]) if args['ou_noise'] else None

    run()

    plot_results()
