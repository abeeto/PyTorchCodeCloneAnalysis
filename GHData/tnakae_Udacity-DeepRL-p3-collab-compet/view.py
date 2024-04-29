import sys
import numpy as np
import pickle
import torch
from unityagents import UnityEnvironment

from PPO import GaussianActorCriticNetwork

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def to_tensor(x, dtype=np.float32):
    return torch.from_numpy(np.array(x).astype(dtype)).to(device)


def get_env_info(env):
    # reset the environment
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    n_agent = len(env_info.agents)
    action_dim = brain.vector_action_space_size
    states = env_info.vector_observations
    state_dim = states.shape[1]

    return n_agent, state_dim, action_dim


def view():
    if sys.platform == "darwin":
        binary_path = "./bin/Tennis.app"
    elif sys.platform == "linux":
        binary_path = "./bin/Tennis_Linux/Tennis.x86_64"
    else:
        binary_path = "./bin/Tennis_Windows_x86_64/Tennis.exe"

    env = UnityEnvironment(file_name=binary_path)
    n_agent, state_dim, action_dim = get_env_info(env)
    model = GaussianActorCriticNetwork(state_dim, action_dim, hiddens=[512, 256])
    model = model.to(device)

    # load best Model
    state_dict = torch.load("bestmodel.pth",
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    # Reset Env
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]
    states = to_tensor(env_info.vector_observations)

    n_step = 5000
    model.eval()
    for step in range(n_step):
        # draw action from model
        actions, _, _, _ = model(states)

        # one step forward
        actions_np = actions.cpu().numpy()
        env_info = env.step(actions_np)[brain_name]
        states = to_tensor(env_info.vector_observations)

if __name__ == "__main__":
    view()
