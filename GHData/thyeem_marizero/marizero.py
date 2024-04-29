import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
import os.path
import random
import pickle
import time
import policy
from collections import deque
from board import Board
from const import Stone, N, CI, H1, H2, H4, LEARNING_RATE, \
                  L2_CONST, GAMMA, N_EPISODE, N_STEPS, \
                  SIZE_DATA, SIZE_BATCH, RATIO_OVERTURN


class Net(nn.Module):
    """ network for both policy p and value function 
    consist of 1 input layer and 2 output layers:
    policy p network -> P(s, a)
    value v network -> v
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CI, H1, (3,3,), 1, 1)
        self.bn1 = nn.BatchNorm2d(H1)
        self.conv2 = nn.Conv2d(H1, H2, (3,3,), 1, 1)
        self.bn2 = nn.BatchNorm2d(H2)
        self.conv3 = nn.Conv2d(H2, H2, (3,3,), 1, 1)
        self.bn3 = nn.BatchNorm2d(H2)
        self.conv4 = nn.Conv2d(H2, H4, (3,3,), 1, 1)
        self.bn4 = nn.BatchNorm2d(H4)
        # self.conv5 = nn.Conv2d(H4, H4, (3,3,), 1, 1)
        # self.bn5 = nn.BatchNorm2d(H4)
        # self.conv6 = nn.Conv2d(H4, H4, (3,3,), 1, 1)
        # self.bn6 = nn.BatchNorm2d(H4)

        self.p_conv = nn.Conv2d(H4, 2, (1,1,), 1, 0)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc1 = nn.Linear(2*N*N, N*N)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.v_conv = nn.Conv2d(H4, 1, (1,1,), 1, 0)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(N*N, 256)
        self.v_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        # x = self.conv5(x)
        # x = F.relu(self.bn5(x))
        # x = self.conv6(x)
        # x = F.relu(self.bn6(x))

        p_x = self.p_conv(x)
        p_x = F.relu(self.p_bn(p_x))
        p_x = p_x.view(-1, 2*N*N)
        p_x = self.p_fc1(p_x)
        p_x = self.logsoftmax(p_x) 

        v_x = self.v_conv(x)
        v_x = F.relu(self.v_bn(v_x))
        v_x = v_x.view(-1, N*N)
        v_x = self.v_fc1(v_x)
        v_x = F.relu(v_x)
        v_x = self.v_fc2(v_x)
        v_x = torch.tanh(v_x)
        return p_x, v_x


def read_state(board):
    """ board -> S
    Defines the input layer S: read the current state from 
    the board given, then return state S of 10 channels
    0 -> turn to play 
    1 -> BLACK stones
    2 -> WHITE stones
    3 -> last move
    4-6 -> enemy's stones captured (one-hot)
    7-9 -> my stones captured (one-hot)
    """
    S = np.zeros([1, CI, N, N])
    S[:,0,:,:] = board.turn
    x, y = board.get_last_move()
    S[:,3,x,y] = 1
    for x in range(N):
        for y in range(N):
            if board.get_stone(x, y) == Stone.BLACK:
                S[:,1,x,y] = 1
            elif board.get_stone(x, y) == Stone.WHITE:
                S[:,2,x,y] = 1

    if board.turn == Stone.BLACK:
        (cap_self, cap_enemy) = (board.scoreB, board.scoreW)
    else:
        (cap_self, cap_enemy) = (board.scoreW, board.scoreB)
    if cap_enemy:
        b2, b1, b0 = [ int(i) for i in f'{cap_enemy-1:03b}' ]
        S[:,4,:,:] = b2
        S[:,5,:,:] = b1
        S[:,6,:,:] = b0
    if cap_self:
        b2, b1, b0 = [ int(i) for i in f'{cap_self-1:03b}' ]
        S[:,7,:,:] = b2
        S[:,8,:,:] = b1
        S[:,9,:,:] = b0
    return S

def xy(move): return move // N, move % N

#---------------------------------------------------------------------

class MariZero(object):
    """ MariZero is a BCF-AI without any human domain knowledge.
    This is the MariAI's next work using neural networks.
    MariAI also was a BCF-AI based on MCTS play rollouts.
    Feel free to meet them all at http://sofimarie.com

    fn and var names based on the following notaion summarized:
    (P(s,-), v) = f_theta(s)
    pi(a|s) = N(s,a)^(1/tau) / Sigma_b N(s,b)^(1/tau)
    a_t = argmax_a ( U(s,a) := Q(s,a) + u(s,a) ) 
    u(s,a) = c_puct * P(s,a) * sqrt(Sigma_b N(s,b)) / (1 + N(s,a))
    N(s,a) = Sigma_i^n 1(s,a,i)
    Q(s,a) = W(s,a) / N(s,a), where W(s,a) := W(s,a) + v

    """
    def __init__(self, game=None):
        """ self.pi -> an instance of policy pi class, TT
        """
        self.init_env()
        self.game = game
        self.pi = policy.TT(self.model)

    def init_env(self):
        self.init_model()
        self.load_episode()
        self.load_selfplays()

    def init_model(self):
        file = self.path_data('model')
        self.model = self.load_model(file) 
        self.init_optim()

    def load_model(self, file=''):
        model = Net()
        if not os.path.isfile(file): return model
        model.load_state_dict(torch.load(file))
        model.eval()
        return model
    
    def save_model(self, file):
        torch.save(self.model.state_dict(), file)

    def update_model(self, overturn=False): 
        file = self.path_data('model')
        if not file: overturn = True
        if overturn: 
            self.save_model(file)
        else:
            self.model = self.load_model(file)
            self.init_optim()

    def init_optim(self):
        self.optim = optim.Adam(self.model.parameters(), 
                                weight_decay=L2_CONST, lr=LEARNING_RATE)
    
    def load_episode(self):
        file = self.path_data('episode')
        if not os.path.isfile(file): self.episode = 1
        else: self.episode = self.load(file)

    def update_episode(self):
        self.episode += 1
        file = self.path_data('episode')
        self.dump(self.episode, file)

    def load_selfplays(self):
        file = self.path_data('plays')
        if os.path.isfile(file): self.data = self.load(file)
        else: self.data = deque(maxlen=SIZE_DATA)

    def save_selfplays(self):
        file = self.path_data('plays')
        if self.episode % 100 == 0: self.dump(self.data, file)

    def path_data(self, name):
        return {
            'model': f'./data/best_model.pt',
            'episode': f'./data/EPISODE',
            'plays': f'./data/SELFPLAYS',
        }.get(name)

    def dump(self, o, file):
        with open(file, 'wb') as f: pickle.dump(o, f)

    def load(self, file):
        with open(file, 'rb') as f: return pickle.load(f)

    def augment_data(self, data):
        """ data augmentation using the symmetry of game board.
        Augments data set by flipping and rotating
        by definition x8 num of data can be produced.
        """
        increment = 0
        for S, pi, z in data:
            _R = [ np.rot90(S, i, axes=(2,3)).copy() for i in range(4) ] 
            _F = [ np.flip(r, 3).copy() for r in _R ]
            _S = _R + _F
            pi = pi.reshape((N,N))
            _R = [ np.rot90(pi, i, axes=(0,1)).copy() for i in range(4) ] 
            _F = [ np.flip(r, 1).copy() for r in _R ]
            _pi =[ x.flatten() for x in _R + _F ]
            _z = np.repeat(z, len(_S))
            self.data.extend(zip(_S, _pi, _z))
            increment += len(_S)
        print(f'data added  {increment:4d}\ttotal {len(self.data):6d}')

    def sample_from_pi(self, pi):
        """ pi(-|s) as [prob] -> best_move
        exploration using Dirichlet noise was not applied.
        Literally sampling, but it depends on the temperature, tau
        when pi is calculated. Converged to the best move as tau -> 0
        """
        best_move = np.random.choice(range(N*N), p=pi)
        return best_move

    def self_play(self):
        """ None -> [ (state S, pi, z), ]
        generates self-play data for training the model
        """
        board = Board()
        _S, _pi, _turn = [], [], []
        self.pi.reset_tree()
        tick = time.time()
        print(f'\nepisode {self.episode:06d}  '
              f'self-play', end='  ', flush=True)
        while True:
            pi = self.pi.fn_pi(board)
            move = self.sample_from_pi(pi)
            self.pi.update_root(move)
            _S.append(read_state(board))
            _pi.append(pi)
            _turn.append(board.whose_turn())

            board.make_move(*xy(move), True)
            winner = board.check_game_end()
            if not winner: continue
            tock = time.time()
            _turn = np.array(_turn)
            _z = np.zeros(len(_turn))
            _z[_turn == winner] = 1
            _z[_turn != winner] = -1
            print(f'{winner == 1 and "black" or "white"} won  '
                  f'{board.moves:3d} moves  {(tock-tick)/60:.2f} mins')
            return zip(_S, _pi, _z)

    def gen_training_data(self, num_games):
        for _ in range(num_games):
            data = self.self_play()
            self.augment_data(data)
            self.update_episode()
            self.save_selfplays()

    def get_training_data(self, batch_size):
        """ get mini-batch from self-play data
        S -> input: board state, from read_state(board)
        pi -> 1-d vector of pi(-|s)
        z -> reward, target of value network
        """
        batch = random.sample(self.data, batch_size)
        S, pi, z = zip(*batch)
        S = torch.cat([ torch.FloatTensor(x) for x in S ], dim=0)
        pi = torch.stack([ torch.FloatTensor(x) for x in pi ])
        z = torch.cat([ torch.FloatTensor([x]) for x in z ], dim=0)
        return S, pi, z
        
    def train(self):
        """
        loss := (z-v)^2 - pi'log(p) + c||theta||^2
        batch from self.data := [ (S, pi, z), ]
        S -> input: board state, from read_state(board)
        z -> reward, target of value network
        v -> output of value network
        cross entropy of pi(a|s) and P(s,a)
        pi -> 1-d vector of pi(-|s)
        log(p) -> output of policy p network
        L2 regularization was considered as L2 penalty in optim
        """
        self.model.train()
        while True:
            self.gen_training_data(N_EPISODE)
            if len(self.data) < SIZE_BATCH: continue
            for i in range(N_STEPS):
                S, pi, z = self.get_training_data(SIZE_BATCH)
                self.optim.zero_grad()
                logP, v = self.model(S)
                loss_v = F.mse_loss(v.view(-1), z)
                loss_P = -torch.mean(torch.sum(pi * logP, 1))
                loss = loss_v + loss_P
                loss.backward()
                self.optim.step()
                print(f'steps {i+1:02d}  loss {loss:.6f}')
            overturn = self.evaluate_model()
            self.update_model(overturn)

    def evaluate_model(self): 
        return True

    def next_move(self, board):
        """ interface responsible for answering game.py module
        """
        if board.moves > 0:
            x, y = board.get_last_move()
            self.pi.update_root(x*N+y)
        pi = self.pi.fn_pi(board)
        best_move = self.sample_from_pi(pi)
        #self.pi.root.print_tree(self.pi.root, cutoff=3)
        return xy(best_move)


if __name__ == '__main__':
    mario = MariZero()
    mario.train()

