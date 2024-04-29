import gym
from Ddpg import *
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('Pendulum-v0')
env.reset()
batch_size = 128
Cheetah = Ddpg(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], actor_hid=[256, 128],
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
        action = Cheetah.run(observation)
        action = Noise.get_action(action, steps*episode)
        next_observation, reward, done, _ = env.step(action)  # take a random action
        rewards += reward
        # env.render()  # To see the chetaah while training uncomment this

        Cheetah.memory.push(observation, action, reward, next_observation, done)
        if len(Cheetah.memory) > batch_size:
            Cheetah.train()

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
        plt.title("Rewards vs Episodes (LunarLanderContinuous)")
        plt.plot(instan_reward, label="Instantenous Reward")
        plt.plot(cumulative_reward, label="Last 5 Reward Mean")
        plt.plot()
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend(loc='best')
        plt.savefig("FigureLunarLanderContinuous" + str(episode + 1) + ".jpeg")
        plt.show()
        try:
            torch.save(Cheetah.Actor.state_dict(), "ActorEpisode" + FileName)
            torch.save(Cheetah.Critic.state_dict(), "CriticEpisode" + FileName)
            torch.save(Cheetah.Target_Actor.state_dict(), "TargetActorEpisode" + FileName)
            torch.save(Cheetah.Target_Critic.state_dict(), "TargetCriticEpisode" + FileName)
        except:
            print(FileName + "could not saved")
            pass

    if rewards >= max(instan_reward):
        try:
            torch.save(Cheetah.Actor.state_dict(), "BestActor.pth")
            torch.save(Cheetah.Critic.state_dict(), "BestCritic.pth")
            torch.save(Cheetah.Target_Actor.state_dict(), "BestTargetActor.pth")
            torch.save(Cheetah.Target_Critic.state_dict(), "BestTargetCritic.pth")
        except:
            print("Bests could not saved")
            pass

plt.title("Rewards vs Episodes (LunarLanderContinuous)")
plt.plot(instan_reward, label="Instantenous Reward")
plt.plot(cumulative_reward, label="Last 5 Reward Mean")
plt.plot()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.savefig("FigureLunarLanderContinuous.jpeg")
plt.show()
env.close()
