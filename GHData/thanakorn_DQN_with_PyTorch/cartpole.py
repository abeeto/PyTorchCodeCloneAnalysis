#%%
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

from itertools import count
from models.dqn import SimpleDQN
from memory.replay_memory import Experience, ReplayMemory
from policy.epsilon_greedy import EpsilonGreedy
from agent import Agent
from env_manager.cartpole_env_manager import CartPoleEnvManager
from utilities.plot import plot
from utilities.tensor import extract_experiences, get_q_next

# Initialization
batch_size = 256
gamma = 0.999
init_epsilon = 1.
min_epsilon = 0.01
epsilon_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.01
num_episodes = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Explore environment
env_manager = CartPoleEnvManager(device)
env_manager.reset()
screen = env_manager.get_processed_screen()
env_manager.close()
plt.figure()
plt.title('Processes Screen')
plt.axis('off')
plt.imshow(screen.squeeze(0).permute(1,2,0))
plt.show()

screen = env_manager.get_state()
plt.figure()
plt.title('Initial State')
plt.axis('off')
plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
plt.show()

env_manager.take_action(1)
env_manager.take_action(1)
screen = env_manager.get_state()
plt.figure()
plt.title('State')
plt.axis('off')
plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
plt.show()

# %% Create instances
env_manager = CartPoleEnvManager(device)
policy = EpsilonGreedy(init_epsilon, min_epsilon, epsilon_decay)
replay_memory = ReplayMemory(memory_size)
_, _, img_height, img_width = env_manager.get_img_dimensions()
policy_net = SimpleDQN(img_height, img_width, env_manager.num_actions()).to(device)
agent = Agent(policy, env_manager.num_actions, device)
target_net = SimpleDQN(img_height, img_width, env_manager.num_actions()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr)

# %% Training
policy_net.train()
episode_durations = []
for ep in range(num_episodes):
    env_manager.reset()
    state = env_manager.get_state()
    
    for timestep in count():
        action = agent.act(state, policy_net)
        reward = env_manager.take_action(action.item())
        next_state = env_manager.get_state()
        replay_memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if len(replay_memory.memory) >= batch_size:
            experiences = replay_memory.sample(batch_size)
            states, actions, rewards, next_states = extract_experiences(experiences)
            q = policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
            q_next = get_q_next(target_net, next_states)
            target = rewards + (gamma * q_next)
            loss = F.mse_loss(q, target.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if env_manager.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break
            
    if ep > 0 and ep % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    env_manager.close()