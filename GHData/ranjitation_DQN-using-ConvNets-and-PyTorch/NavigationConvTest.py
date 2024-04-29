from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import torch

from model import QNetwork
from agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = UnityEnvironment(file_name="Banana.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents in the environment
n_agents = len(env_info.agents)
# print('Number of agents: ', n_agents)

# number of actions
action_size = brain.vector_action_space_size
# print('Action Size: ', action_size)

# examine the state space
state = np.squeeze(env_info.visual_observations[0])
state_size = state.shape

stack_size = 4

# Initialize deque with zero-images. One array for each image
stacked_frames = deque([np.zeros((3, 84, 84), dtype=np.int) for _ in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):

    frame = np.squeeze(state)

    if is_new_episode:
        # Clear our stacked frames
        stacked_frames = deque([np.zeros((3, 84, 84), dtype=np.int) for _ in range(stack_size)], maxlen=4)

        # Because we're in a new episode,copy the same frame 4 times
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=0)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=0)

    return stacked_state, stacked_frames


agent = Agent(action_size=action_size)

agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

score = 0.0

for i_episode in range(10):
    env_info = env.reset(train_mode=False)[brain_name]
    state = np.reshape(np.squeeze(env_info.visual_observations[0]), [3, 84, 84])             
    stacked_state, stacked_frames = stack_frames(stacked_frames, state, is_new_episode=True) 
    while True:
        action = agent.act(stacked_state, epsilon=0)
        env_info = env.step(np.int(action))[brain_name]
        next_state = np.reshape(np.squeeze(torch.from_numpy(env_info.visual_observations[0])), [3, 84, 84])
        stacked_next_state, stacked_frames = stack_frames(stacked_frames, next_state, is_new_episode=False)
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        stacked_state = stacked_next_state
        score += reward
        if done:
            print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score, end=""))
            break

env.close()
