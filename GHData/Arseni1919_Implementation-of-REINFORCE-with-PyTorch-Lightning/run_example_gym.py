from IMPORTS import *
env = gym.make('procgen:procgen-starpilot-v0')
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('Tutankham-ram-v0')

# env = gym.make('CarRacing-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('Acrobot-v1')
obs = env.reset()
done = False
for i in range(30000):
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()
    # print(rew)
    if done:
        print('done')
        obs = env.reset()
env.close()