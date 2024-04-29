import gym
from Ddpg import *
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('MountainCarContinuous-v0')
batch_size = 128
Driver = Ddpg(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], actor_hid=[256, 128],
               critic_hid=[256, 128], actor_lr=1e-4, critic_lr=1e-3,
               gamma=0.99, tau=1e-2, replay_size=100000, batch_size=batch_size)
Noise = OUNoise(env.action_space, min_sigma=0)
instan_reward = []
cumulative_reward = []
Noise.reset()

for episode in range(100):
    observation = env.reset()  # observation = state
    Noise.reset()
    rewards = 0

    for steps in range(1000):
        action = Driver.run(observation)
        action = Noise.get_action(action, steps*episode)
        next_observation, reward, done, _ = env.step(action)  # take a random action
        rewards += reward
        # env.render()  # To see the chetaah while training uncomment this

        Driver.memory.push(observation, action, reward, next_observation, done)
        if len(Driver.memory) > batch_size:
            Driver.train()

        observation = next_observation

        if done and episode % 10 == 0:
            instan_reward.append(rewards)
            cumulative_reward.append(np.mean(instan_reward[-5:]))
            print("Episode " + str(episode) + " has been completed with:")
            print("Reward: " + str(instan_reward[len(instan_reward) - 1]))
            print("Avarage Cumulative Reward: " + str(cumulative_reward[len(cumulative_reward) - 1]))
            print("At Step: " + str(steps))
            print()
            break

    instan_reward.append(rewards)
    cumulative_reward.append(np.mean(instan_reward[-5:]))

    if episode % 100 == 99 and episode > 1:
        FileName = str(episode + 1) + ".pth"
        plt.title("Rewards vs Episodes (MountainCarContinuous)")
        plt.plot(instan_reward, label="Instantenous Reward")
        plt.plot(cumulative_reward, label="Last 5 Reward Mean")
        plt.plot()
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend(loc='best')
        plt.savefig("MountainCarContinuous" + str(episode + 1) + ".jpeg")
        plt.show()
        try:
            torch.save(Driver.Actor.state_dict(), "CarActorEpisode" + FileName)
            torch.save(Driver.Critic.state_dict(), "CarCriticEpisode" + FileName)
            torch.save(Driver.Target_Actor.state_dict(), "CarTargetActorEpisode" + FileName)
            torch.save(Driver.Target_Critic.state_dict(), "CarTargetCriticEpisode" + FileName)
        except:
            print(FileName + "could not saved")
            pass

    if rewards >= max(instan_reward):
        try:
            torch.save(Driver.Actor.state_dict(), "CarBestActor.pth")
            torch.save(Driver.Critic.state_dict(), "CarBestCritic.pth")
            torch.save(Driver.Target_Actor.state_dict(), "CarBestTargetActor.pth")
            torch.save(Driver.Target_Critic.state_dict(), "CarBestTargetCritic.pth")
        except:
            print("Bests could not saved")
            pass

plt.title("Rewards vs Episodes (MountainCarContinuous)")
plt.plot(instan_reward, label="Instantenous Reward")
plt.plot(cumulative_reward, label="Last 5 Reward Mean")
plt.plot()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.savefig("MountainCarContinuous.jpeg")
plt.show()
env.close()
