from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN


agent = DQN(MlpPolicy, 'CartPole-v0', verbose=1, tensorboard_log='runs')

agent.learn(40000)
