from gomoku import GomokuEnv
from time import sleep
from datetime import datetime
from ai import BOARD_SIZE, WuziGo, coord_to_action
import numpy as np
player_color = 'black'

# default 'beginner' level opponent policy 'random'
debug = False
train = False
env = GomokuEnv(player_color=player_color,
                opponent='beginner', board_size=BOARD_SIZE)
ai = WuziGo('player26')
ai.load()
EPOCHS = 1
STEPS = 100
for epoch in range(EPOCHS):
    loss_array = []
    win, draw, lose = 0, 0, 0
    start = datetime.now().timestamp()
    for step in range(STEPS):
        env.reset()
        observation = env.state.board.board_state
        observation = np.where(observation == 2, -1, observation)
        while True:
            action = ai.play(observation)
            observation, reward, done, info = env.step(action)
            if debug:
                env.render()
                sleep(1)
            observation = np.where(observation == 2, -1, observation)
            coord = env.state.board.last_coord
            if reward == 1 or done:
                break
            action = coord_to_action(*coord)
            ai.observe(action, observation)
            if done:
                break
        if train:
            loss = ai.reward(reward)
            loss_array.append(loss)
        if reward == 1:
            win += 1
        elif reward == -1:
            lose += 1
        else:
            draw += 1
        if debug:
            print('win:', win, 'drar:', draw, 'lose:', lose)
    end = datetime.now().timestamp()
    print('epoch:', epoch, 'takes', end - start,)
    print('mean loss', np.mean(
        loss_array), 'win:', win, 'drar:', draw, 'lose:', lose)
    if train:
        ai.save(epoch)
