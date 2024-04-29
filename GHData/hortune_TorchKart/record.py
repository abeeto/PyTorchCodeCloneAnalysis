from pynput.keyboard import Key, Listener
import gym, gym_mupen64plus
import sys
import numpy as np
import argparse

class KeyboardController:
    def __init__(self, degree=80):
        self.action = [0, 0, 0, 0, 0]
        self.degree = degree
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.start()
    def on_press(self, key):
        if key == Key.space:
            self.action[2] = 1
        if key == Key.ctrl:
            self.action[3] = 1
        if key == Key.shift:
            self.action[4] = 1
        if key == Key.right:
            self.action[0] = self.degree
        if key == Key.left:
            self.action[0] = -1 * self.degree
    def on_release(self, key):
        if key == Key.space:
            self.action[2] = 0
        if key == Key.ctrl:
            self.action[3] = 0
        if key == Key.shift:
            self.action[4] = 0
        if key == Key.right:
            self.action[0]  = 0
        if key == Key.left:
            self.action[0]  = 0
    def start(self):
        self.listener.start()
    def stop(self):
        self.listener.stop()

def process_frame(frame):
    #frame = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
    frame = frame.astype(np.uint8)[::2, ::2,:]
    return frame

env_list = [ 
    'Mario-Kart-Luigi-Raceway-v0',
    'Mario-Kart-Moo-Moo-Farm-v0',
    'Mario-Kart-Koopa-Troopa-Beach-v0',
    'Mario-Kart-Kalimari-Desert-v0',
    'Mario-Kart-Toads-Turnpike-v0',
    'Mario-Kart-Frappe-Snowland-v0',
    'Mario-Kart-Choco-Mountain-v0',
    'Mario-Kart-Mario-Raceway-v0',
    'Mario-Kart-Wario-Stadium-v0',
    'Mario-Kart-Sherbet-Land-v0',
    'Mario-Kart-Royal-Raceway-v0',
    'Mario-Kart-Bowsers-Castle-v0',
    'Mario-Kart-DKs-Jungle-Parkway-v0',
    'Mario-Kart-Yoshi-Valley-v0',
    'Mario-Kart-BansheeBoardwalk-v0',
    'Mario-Kart-Rainbow-Road-v0']
if __name__ == '__main__':
    env = gym.make(env_list[0])
    controller = KeyboardController()
    env.reset()
    done = False
    total_reward = 0
    x_data, y_data = [], []
    try:
        turn = False
        index = 0
        while not done:
            env.render()
            action = controller.action[:]
            observation, reward, done, _ = env.step(action)
            if index % 5 == 0:
                x_data.append(process_frame(observation))
                y_data.append(action)
            total_reward += reward if reward < 0 else reward * 10
            print('action:{}, reward:{}'.format(action, total_reward))
            index+=1
            if reward == 100:
                if turn:
                    break
                turn = True
    except KeyboardInterrupt:
        pass
    print('Game finish.')
    print('Save data.')
    np.save('{}_x.npy'.format(sys.argv[1]), np.array(x_data))
    np.save('{}_y.npy'.format(sys.argv[1]), np.array(y_data))
    controller.stop()
    env.close()

