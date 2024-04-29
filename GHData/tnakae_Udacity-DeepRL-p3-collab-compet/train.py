import sys
import numpy as np
import pickle
import torch
from unityagents import UnityEnvironment

from PPO import GaussianActorCriticNetwork
from PPO import PPOAgent


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


def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if sys.platform == "darwin":
        binary_path = "./bin/Tennis.app"
    elif sys.platform == "linux":
        binary_path = "./bin/Tennis_Linux_NoVis/Reacher.x86_64"
    else:
        binary_path = "./bin/Tennis_Windows_x86_64/Reacher.exe"

    env = UnityEnvironment(file_name=binary_path)
    n_agent, state_dim, action_dim = get_env_info(env)
    model = GaussianActorCriticNetwork(state_dim, action_dim, hiddens=[512, 256])
    model = model.to(device)
    agent = PPOAgent(env, model, tmax=1024, n_epoch=20,
                     batch_size=128, eps=0.1, device=device)

    n_step = 2000
    n_episodes = 0
    for step in range(n_step):
        agent.step()
        scores = agent.scores_by_episode
        if n_episodes < len(scores):
            n_episodes = len(scores)
            print(f" episode #{n_episodes} : score = {scores[-1]:.2f}", end="")
            if 100 <= n_episodes:
                rewards_ma = np.mean(scores[-100:])
                print(f", mean score of last 100 episodes = {rewards_ma:.2f}")
                if .5 <= rewards_ma:
                    torch.save(model.state_dict(), "bestmodel.pth")
                    with open('rewards.pickle', 'wb') as fp:
                        pickle.dump(scores, fp)
                    print("\n ==== Achieved criteria! Model is saved.")
                    break
            else:
                print()

        sys.stdout.flush()

    print("Finished.")

if __name__ == "__main__":
    train()
