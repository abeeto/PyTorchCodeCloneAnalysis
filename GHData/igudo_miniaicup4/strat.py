import json
import numpy as np
import torch
from torch import nn
from autolearner import Net
config = json.loads(input())
score = 0


def make_first_ls(ls):
    learning_state = {}
    cells = [0.2 for _ in range(31*31)]
    for cell_i in range(len(cells)):
        learning_state[f'cell{cell_i}'] = cells[cell_i]
    return learning_state


def make_normal_coord(coord):
    return coord // config['params']['width']


def edit_cells_by_player(cells, player):
    for my_cell in player['territory']:
        x = make_normal_coord(my_cell[0])
        y = make_normal_coord(my_cell[1])
        cells[x * y] = 1
    if player.get('lines', False):
        for line in player['lines']:
            x = make_normal_coord(line[0])
            y = make_normal_coord(line[1])
            cells[x * y] = 2
    x = make_normal_coord(player['position'][0])
    y = make_normal_coord(player['position'][1])
    cells[x * y] = -2
    return cells


def make_ls(lstate):
    global score
    commands_int = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
    ls = lstate['params']['players']['i']

    cells = [0.5 for _ in range(31*31)]

    for my_cell in ls['territory']:
        x = make_normal_coord(my_cell[0])
        y = make_normal_coord(my_cell[1])
        cells[x * y] = 1.5
    if ls.get('lines', False):
        for line in ls['lines']:
            x = make_normal_coord(line[0])
            y = make_normal_coord(line[1])
            cells[x * y] = -0.5
    x = make_normal_coord(ls['position'][0])
    y = make_normal_coord(ls['position'][1])
    cells[x * y] = 0

    for pl in lstate['params']['players'].keys():
        if pl == 'i':
            continue
        cells = edit_cells_by_player(cells, lstate['params']['players'][pl])

    learning_state = {}

    for cell_i in range(len(cells)):
        learning_state[f'cell{cell_i}'] = cells[cell_i]

    reward = ls['score'] - (score)
    score = ls['score']
    return learning_state, reward


model = Net()
try:
    model.load_state_dict(torch.load('params/param19'))
    model.eval()
except FileNotFoundError:
    pass

activation = nn.Softmax(dim=0)
states, actions = [], []
total_reward = 0


def get_command(state):
    ls, reward = make_ls(state)
    ls = list(ls.values())

    action = int(torch.argmax(model.predict(ls)))

    return action


s = None
tick = 1
last_territory = []

while True:
    try:
        main_s = json.loads(input())
        tick = main_s.get('params', {}).get('tick_num', 1)
        new_s, r = make_ls(main_s)
        new_s = new_s
    except:
        break

    if s is None:
        s = list(make_first_ls(new_s).values())
    new_s = list(new_s.values())

    territory = main_s['params']['players']['i']['territory']
    if last_territory:
        if len(territory) > len(last_territory):
            r += 50
    last_territory = territory

    commands = ['left', 'right', 'up', 'down']

    s_v = torch.FloatTensor(np.array(new_s).reshape([1, 1, 31, 31]))
    act_probs_v = activation(model(s_v)[0])
    act_probs = act_probs_v.data.numpy()
    a = np.random.choice(commands, p=act_probs_v.detach().numpy())
    states.append(s)
    actions.append(commands.index(a))
    total_reward += r

    s = new_s

    pos = main_s['params']['players']['i']['position']

    print(json.dumps({"command": a, 'debug': str(new_s)}))

if tick == 0:
    tick = 1
if tick < 50:
    total_reward = -300
elif tick <= 120:
    total_reward = total_reward - tick
total_reward = total_reward - (20/tick)
with open('log', 'w') as f:
    f.write(
        str(states)+'\n'
        + str(actions)+'\n'
        + str(total_reward)
    )
