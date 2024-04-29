import gym
import numpy as np

def relative_position():
    i = 0
    while True:
        idx = i % 3
        applied_action = np.zeros(3)
        observation = env.reset()
        env.setRealTimeSimulation()
        print(env.action_space.low[idx], env.action_space.high[idx])
        input("Start")
        for action in range(10):
            applied_action[idx] = env.action_space.high[idx] / 10
            print(applied_action)
            _,_,_,_ = env.step(applied_action)

        input("+ve end")
        for action in range(10):
            applied_action[idx] = - env.action_space.high[idx] / 10
            print(applied_action)
            _,_,_,_ = env.step(applied_action)


        input("Middle")
        for action in range(10):
            applied_action[idx] = env.action_space.low[idx] / 10
            print(applied_action)
            _,_,_,_ = env.step(applied_action)

        input("-ve End")
        for action in range(10):
            applied_action[idx] = - env.action_space.low[idx] / 10
            print(applied_action)
            _,_,_,_ = env.step(applied_action)

        i += 1

def absolute_position():
    i = 0
    while True:
        idx = i % 3
        applied_action = np.zeros(3)
        observation = env.reset()
        env.setRealTimeSimulation()
        print(env.action_space.low[idx], env.action_space.high[idx])
        input("Start")
        for action in np.linspace(0, env.action_space.high[idx], 10):
            applied_action[idx] = action
            print(applied_action)
            _,_,_,_ = env.step(applied_action)

        input("+ve end")
        for action in np.linspace(env.action_space.high[idx], 0, 10):
            applied_action[idx] = -action
            print(applied_action)
            _,_,_,_ = env.step(applied_action)


        input("Middle")
        for action in np.linspace(0, env.action_space.low[idx], 10):
            applied_action[idx] = action
            print(applied_action)
            _,_,_,_ = env.step(applied_action)

        input("-ve End")
        for action in np.linspace(env.action_space.high[idx], 0, 10):
            applied_action[idx] = -action
            print(applied_action)
            _,_,_,_ = env.step(applied_action)

        i += 1


env = gym.make('gym_finger:Finger-v0')
absolute_position()
