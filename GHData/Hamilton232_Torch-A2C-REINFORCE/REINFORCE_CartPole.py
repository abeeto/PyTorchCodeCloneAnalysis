import matplotlib.pyplot as plt
import os
import numpy as np

import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

import gym

class MLP(torch.nn.Module):

    def __init__(self, state_num, H, actions_num):
        super(MLP, self).__init__()
        self.state_num = state_num
        self.l1 = torch.nn.Linear(state_num, H)
        self.l2 = torch.nn.Linear(H, H)
        self.actor = torch.nn.Linear(H, actions_num)

    def forward(self, x):
        flat = x.view(-1, self.state_num)
        hidden = F.relu(self.l1(flat))
        hidden = F.relu(self.l2(hidden))
        act = self.actor(hidden)
        return F.softmax(act), F.log_softmax(act)

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def select_action(state, network):
    state = torch.from_numpy(state).float()
    prob, log_prob = network(state)

    m = Categorical(prob)
    action = m.sample()

    return action.item(), m.log_prob(action), prob, log_prob




def main():
    # -------------------------------- Environment --------------------------------
    environment = gym.make("CartPole-v0")
    # ---------------------------------Hyper Parameters--------------------------------Change to be parsed in for real tests
    # Discount factor. Model is not very sensitive to this value.
    N_episodes = 100
    GAMMA = .99
    ACTION_DIM = 2  # 4
    STATE_DIM = 4  # np.size(state)
    H = 20
    LR = 0.01
    # --------------------------------Neural Networks--------------------------------
    network = MLP(STATE_DIM, H, ACTION_DIM)
    network_optim = optim.Adam(network.parameters(), lr=LR)
    # -------------------------------- Training --------------------------------
    save = 'REINFORCE' + str(N_episodes) + '_'
    cnt = 0
    episode = 0

    # Arrays for results
    ep_rewards_avg = []

    # Losses
    loss_policy = []

    while episode < N_episodes:
        cnt += 1
        episode += 1
        q_batch = []
        chosen_log_batch = []

        ep_rewards = []

        for i in range(10):

            rewards = []
            actions_log = []
            state = environment.reset()
            done = False
            while done == False:
                action, action_log, prob, log_prob = select_action(state, network)
                # Step the environment
                next_state, reward, done, meta = environment.step(action)
                rewards.append(reward)
                actions_log.append(action_log)
                state = next_state
            ep_rewards.append(np.sum(rewards))
            q_sa = discount_rewards(rewards)

            q_batch.extend(q_sa)
            chosen_log_batch.extend(actions_log)

        mean_reward = np.mean(ep_rewards)
        ep_rewards_avg.append(mean_reward)

        tmp = torch.Tensor(q_batch)
        tmp2 = torch.stack(chosen_log_batch,1)
        g = tmp*tmp2
        policy_loss = - torch.mean(g)

        network_optim.zero_grad()
        policy_loss.backward()
        network_optim.step()


        if cnt % 50:
            print(episode, ":", mean_reward)
        if cnt % 3:
            loss_policy.append(policy_loss.data.numpy())

    torch.save(network.state_dict(), os.path.join(save + 'A2C_network.torch'))
    np.save(os.path.join(save + 'ep_rewards_avg'), ep_rewards_avg)
    np.save(os.path.join(save + 'loss_policy'), loss_policy)

    plt.figure()
    plt.plot(ep_rewards_avg)
    plt.ylabel('Avg Rewards')
    plt.xlabel('Episodes')
    plt.show()

if __name__ == '__main__':
    main()
