from ai import BOARD_SIZE, WuziGo, coord_to_action
from gomoku import GomokuState, Board
from util import gomoku_util
import numpy as np
from time import sleep
from datetime import datetime
from enum import IntEnum
import random
from os import listdir
from os.path import isfile, join


class TrainMode(IntEnum):
    NO = 0
    PLAYER1 = 1
    PLAYER2 = 2
    BOTH = 3


def check_wins(state):
    done = state.board.is_terminal()
    if not done:
        return False, 0
    # Check Fianl wins
    exist, win_color = gomoku_util.check_five_in_row(
        state.board.board_state)  # 'empty', 'black', 'white'
    reward = 0.
    if win_color == "empty":  # draw
        reward = 0.
    else:
        # check if player_color is the win_color
        player_wins = ('black' == win_color)
        reward = 1. if player_wins else -1.
    return True, reward


def ai_compete(player1, player2, view_mode=False):
    state = GomokuState(
        Board(BOARD_SIZE), 'black')  # Black Plays First
    players = {
        'black': player1,
        'white': player2
    }
    oppos = {
        'black': player2,
        'white': player1
    }
    done = False
    while not done:
        observation = state.board.board_state
        player = players[state.color]
        action = player.play(observation)
        oppos[state.color].observe(action, observation)
        state = state.act(action)
        done, reward = check_wins(state)
        if view_mode:
            print(state.board)
            sleep(1)
    return reward


def train_epoch(player1, player2, max_steps=100, max_win_combo=10, train_mode=TrainMode.PLAYER1, view_mode=False):
    win = 0
    win_combo = 0
    curr_max_win_combo = 0
    start_time = datetime.now()
    for step in range(max_steps):
        if step % 2:
            reward = ai_compete(player2, player1, view_mode)
            reward = -reward
        else:
            reward = ai_compete(player1, player2, view_mode)
        if reward > 0:
            win += 1
            win_combo += 1
            curr_max_win_combo = max(curr_max_win_combo, win_combo)
        elif reward < 0:
            win_combo = 0
        if win_combo >= max_win_combo:
            break
        if view_mode:
            print('step', step, 'reward', reward)
        if train_mode & TrainMode.PLAYER1:
            player1.reward(reward)
        if train_mode & TrainMode.PLAYER2:
            player2.reward(reward)
        if step % 100 == 99:
            print('steps', step + 1, 'win', win / step,
                  'curr max win combo', curr_max_win_combo)
    duration = datetime.now() - start_time
    return win, step, duration


def primary_train(player, max_epochs=50, max_steps=100, max_win_combo=10):
    oppo = WuziGo()
    for epoch in range(max_epochs):
        oppo.copy(player)
        win, step, duration = train_epoch(
            player, oppo, max_steps, max_win_combo)
        print('epoch', epoch, 'loss',
              'win', win / step, 'steps', step, 'duration', duration)
        if step + 1 >= max_steps and win < max_steps / 2:
            oppo.save()
            return
    player.save()


def complete_ais_epoch(player1, player2, max_steps=100, view_mode=False):
    win, step, duration = train_epoch(
        player1, player2, max_steps, max_win_combo=max_steps, train_mode=TrainMode.NO, view_mode=view_mode)
    print('win', win / step, 'steps', step, 'duration', duration)


def train_against_trainer(new_ai, ai_pool, evolve_rate=0.55, max_epochs=10, max_steps=200):
    for epoch in range(max_epochs):
        oppo = random.choice(ai_pool)
        win, step, duration = train_epoch(
            new_ai, oppo, max_steps, max_steps)
        win_rate = win / (step + 1)
        print('ai', new_ai.name, 'epoch', epoch, 'loss',
              'win', win_rate, 'steps', (step + 1), 'duration', duration)
        if win_rate > evolve_rate:
            break
    return win_rate


def rolling_training(num_train, ai_size=20, evolve_rate=0.55, max_attempts=5):
    ai_max_num = getAIMaxNumber()
    ai_pool = loadAIs(ai_size)
    for i in range(ai_max_num + 1, ai_max_num + num_train + 1):
        win_rate = 0
        for j in range(max_attempts):
            new_ai = WuziGo('player' + str(i))
            win_rate = train_against_trainer(new_ai, ai_pool, evolve_rate)
            print('tmp ai', new_ai.name, 'attempt', j, 'win rate', win_rate)
            if win_rate > evolve_rate:
                print('get an evolved ai')
                break
        new_ai.save()
        ai_pool = ai_pool[1:] + [new_ai]
        print('finished training ai', new_ai.name, 'win rate', win_rate)


def getAIMaxNumber():
    ai_path = 'weights/trainers/'
    ai_state_files = [f for f in listdir(ai_path) if isfile(join(ai_path, f))]
    ai_numbers = [int(s.replace('player', '').replace('.pt', ''))
                  for s in ai_state_files]
    return max(ai_numbers + [0])


def loadAIs(ai_size=20):
    max_ai_number = getAIMaxNumber()
    ai_pool = []
    for i in range(max(0, max_ai_number - ai_size + 1), max_ai_number + 1):
        ai = WuziGo(f'player{i}')
        ai.load()
        ai_pool.append(ai)
    for i in range(max_ai_number + 1, ai_size):
        ai = WuziGo(f'player{i}')
        primary_train(ai)
        ai.save()
        ai_pool.append(ai)
    print('getting ais', ','.join([ai.name for ai in ai_pool]))
    return ai_pool


if __name__ == "__main__":
    rolling_training(100)
    # ai1 = WuziGo('player21')
    # ai1.load()
    # ai2 = WuziGo('player22')
    # ai2.load()
    # complete_ais_epoch(ai1, ai2, view_mode=True)
    # for player in range(10):
    #     name = f'player{player}'
    #     ai = WuziGo(name)
    #     print('training', name)
    #     primary_train(ai, max_epochs=10, max_steps=300, max_win_combo=8)
