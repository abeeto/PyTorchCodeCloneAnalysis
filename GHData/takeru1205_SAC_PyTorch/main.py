import gym
import torch

from buffer import ReplayBuffer
from model import Actor
gym.logger.set_level(40)

num_episode = 5

env = gym.make('Pendulum-v0')
buffer = ReplayBuffer(max_size=100)
actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])

for e in range(num_episode):
    cumulative_reward = 0
    state = env.reset()
    for i in range(env.spec.max_episode_steps):
        action = actor(torch.FloatTensor(state)).detach().numpy()

        next_state, reward, done, info = env.step(action * env.action_space.high[0])
        buffer.add([state, next_state, reward, done])

        state = next_state

        cumulative_reward += reward
        
    print(f'Episode: {e:>3}, Reward: {cumulative_reward:>8.2f}')

print(len(buffer))

