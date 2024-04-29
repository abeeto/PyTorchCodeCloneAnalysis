from gomoku import GomokuEnv
from time import sleep
from datetime import datetime
from ai_w3 import BOARD_SIZE, WuziGo, coord_to_action
import numpy as np
player_color = 'black'

# default 'beginner' level opponent policy 'random'
debug = False
train = True
env = GomokuEnv(player_color=player_color,
                opponent='beginner', board_size=BOARD_SIZE)
ai = WuziGo()
ai.load()

env.reset()
observation = env.state.board.board_state
observation = np.where(observation == 2, -1, observation)
while True:
    action = ai.play(observation)
    observation, reward, done, info = env.step(action)
    observation = np.where(observation == 2, -1, observation)
    coord = env.state.board.last_coord
    if reward == 1 or done:
        break
    action = coord_to_action(*coord)
    ai.observe(action, observation)
    env.render()
    sleep(1)
    if done:
        break
env.render()
