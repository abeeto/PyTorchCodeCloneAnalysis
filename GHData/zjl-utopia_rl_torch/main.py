import torch.nn as nn
import torch.nn.functional as F
import gym

from agents.DQNAgent import DQNAgent


env = gym.make('CartPole-v0')   # 'CartPole-v0'
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 2 actions
N_STATES = env.observation_space.shape[0]  # 4 states
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # confirm the shape


class Net(nn.Module):
    def __init__(self, n_state, n_action):
        super(Net, self).__init__()
        # Define the structure of fully connected network
        self.fc1 = nn.Linear(n_state, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, n_action)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# Dueling DQN Network
# class Net(nn.Module):
#     def __init__(self, n_state, n_action):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(n_state, 10)
#         self.fc1.weight.data.normal_(0, 0.1)
#         self.out1 = nn.Linear(10, 1)
#         self.out1.weight.data.normal_(0, 0.1)
#
#         self.fc2 = nn.Linear(n_state, 10)
#         self.fc2.weight.data.normal_(0, 0.1)
#         self.out2 = nn.Linear(10, n_action)
#         self.out2.weight.data.normal_(0, 0.1)
#
#     def forward(self, x):
#         x1 = self.fc1(x)
#         x1 = F.relu(x1)
#
#         x2 = self.fc2(x)
#         x2 = F.relu(x2)
#
#         y1 = self.out1(x1)
#         y2 = self.out2(x2)
#         x3 = y1 + y2 - y2.mean()
#         actions_value = x3
#         return actions_value


'''
--------------Procedures of DQN Algorithm------------------
'''
# create the object of DQN class
agent = DQNAgent(n_state=N_STATES, n_action=N_ACTIONS, action_shape=ENV_A_SHAPE, network=Net,
                 use_target_net=True)

# Start training
print("\nCollecting experience...")
for i_episode in range(400):
    # play 400 episodes of cartpole game
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        # take action based on the current state
        a = agent.select_action(s)
        # obtain the reward and next state and some other information
        s_, r, done, info = env.step(a)

        # modify the reward based on the environment state
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # store the transitions of states
        agent.store_transition(s, a, r, s_, done)

        ep_r += r
        # if the experience repaly buffer is filled, DQN begins to learn or update
        # its parameters.
        if agent.ready_to_learn():
            agent.learn()
            if done:
                print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))

        if done:
            # if game is over, then skip the while loop.
            break
        # use next state to update the current state.
        s = s_
