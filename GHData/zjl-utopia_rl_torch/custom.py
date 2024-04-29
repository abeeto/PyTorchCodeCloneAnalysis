import gym


# env = gym.make('Pong-v0')
# env = gym.make('gym_custom:GridWorld-v0')
env = gym.make('gym_custom:Snake-v0')
env = env.unwrapped
env.reset()
print(env.action_space.n)
print(env.state)
while True:
    env.render()
# env.close()
