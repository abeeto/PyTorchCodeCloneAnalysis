import gym
from tensorboardX import SummaryWriter
from net_model import Net
import torch.nn as nn
import torch.optim as optim
from batch_helpers import iterate_batches, filter_batch

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


def main():
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    env = gym.wrappers.Monitor(env, directory="mon", force=True)
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        create_tensorboard_report(iter_no, loss_v, reward_b, reward_m, writer)

        if reward_m > 199:
            print("Solved!")
            break

    writer.close()


def create_tensorboard_report(iter_no, loss_v, reward_b, reward_m, writer):
    print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
        iter_no, loss_v.item(), reward_m, reward_b))
    writer.add_scalar("loss", loss_v.item(), iter_no)
    writer.add_scalar("reward_bound", reward_b, iter_no)
    writer.add_scalar("reward_mean", reward_m, iter_no)


if __name__ == '__main__':
    main()
