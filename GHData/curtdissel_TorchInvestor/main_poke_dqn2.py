import numpy as np
from dqn_agent import PokeDQNAgent
from utils import make_env, plot_learning_curve
import random

import json
class ReplayEnv():
    def __init__(self):
        with open('replay_json_data.json') as json_file:
            self.replay_data = json.load(json_file)
        self.replay_index = 0

    def reset(self):
        data = self.replay_data[self.replay_index]
        self.replay_index += 1
        self.replay_index = self.replay_index % len(self.replay_data)
        return data["obs"], data["reward"], data["done"], data["info"]

    def step(self, action):
        data = self.replay_data[self.replay_index]
        self.replay_index += 1
        self.replay_index = self.replay_index % len(self.replay_data)
        return data["obs"], data["reward"], data["done"], data["info"]

env = ReplayEnv()
config = {}
state = env.reset()
action2 = None
rewards = 0
steps = 0
sample_obs = state
sample_obs = sample_obs[0]
print('sample_obs:\n')
print(sample_obs)
player='p1'
all_rewards = []
target = 0


if __name__ == '__main__':
#    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 500

    agent = PokeDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(641,), n_actions=14, mem_size=50000, eps_min=0.1, batch_size=32, replace=1000, eps_dec=1e-5, chkpt_dir='models/', algo='PokeDQNAgent', env_name='PokeEnv-v4')
    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' +\
        '_' + str(n_games) + 'games'

    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        score = 0
        observation, _, _, _ = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()

            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score: ', score, 'average score %.1f best score %.1f epsiolon %.2f' % (avg_score, best_score, agent.epsilon), 'steps ', n_steps)
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    plot_learning_curve(steps_array, scores, eps_history, figure_file)
