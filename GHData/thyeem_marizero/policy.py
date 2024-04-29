import math
import numpy as np
import torch
import marizero as mario
from copy import deepcopy
from collections import defaultdict
from const import N, C_PUCT, N_SEARCH

def xy(move): return move // N, move % N

def softmax(x):
    p = np.exp(x-np.max(x))
    p /= p.sum(axis=0)
    return p


class FakeNode(object):
    """ 
    for convenience, here introduced FakeNode. 
    root.prev != None anymore, root node is just a normal node.
    FakeNode.prev := None, and root.prev := FakeNode 
    Thus, root.prev.prev == None.
    [FakeNode] -> [root] -> [children...]

    """
    def __init__(self):
        self.prev = None
        self.next_W = defaultdict(float)
        self.next_P = defaultdict(float)
        self.next_N = defaultdict(float)
        self.N = 0


class Node(object):
    """ definition of node used in Monte-Carlo search for policy pi
    """
    def __init__(self, move=None, prev=None):
        self.move = move
        self.is_expanded = False
        self.prev = prev
        self.next = {}
        self.next_P = np.zeros([N*N], dtype=np.float32)
        self.next_W = np.zeros([N*N], dtype=np.float32)
        self.next_N = np.zeros([N*N], dtype=np.float32)

    @property
    def Q(self):
        return self.W / (self.N + 1)

    @property
    def N(self):
        return self.prev.next_N[self.move]

    @N.setter
    def N(self, value):
        self.prev.next_N[self.move] = value

    @property
    def W(self):
        return self.prev.next_W[self.move]

    @W.setter
    def W(self, value):
        self.prev.next_W[self.move] = value

    def next_Q(self):
        return self.next_W / (self.next_N + 1)

    def next_u(self):
        return C_PUCT * math.sqrt(self.N) * \
               (self.next_P / (self.next_N + 1))

    def best_next(self):
        """ Upper-confidence bound on Q-value
        return argmax_a [ U(s,a) := Q(s,a) + u(s,a) ]
        """
        return np.argmax(self.next_Q() + self.next_u())
    
    def select(self, board):
        node = self
        while node.is_expanded:
            move = node.best_next()
            if board.is_illegal_move(*xy(move)):
                node.next_W[move] = -1e4
                continue
            board.make_move(*xy(move), True)
            node = node.add_child(move)
        return node

    def add_child(self, move):
        if not move in self.next:
            self.next[move] = Node(move, self)
        return self.next[move]

    def expand(self, P):
        self.is_expanded = True
        self.next_P = P

    def backup(self, v):
        node = self
        while node.prev is not None:
            node.N += 1
            node.W += v
            v *= -1
            node = node.prev

    def print_tree(self, node, indent=2, cutoff=None):
        """ recursively dumps node-tree
        usage: node.print_tree(node, cutoff=5)
        """
        x, y = node.move and (xy(node.move)) or (-1, -1)
        N_ = node.N
        P_ = node.prev.next_P[node.move]
        Q_ = node.Q
        u_ = C_PUCT * math.sqrt(node.prev.N) * P_ / (N_ + 1)
        U_ = Q_ + u_
        print(f'{" "*indent} ({x:2d},{y:2d})  N {N_:6.0f}  U {U_:6.4f}  '
              f'Q {Q_:6.4f}  u {u_:6.4f}  P {P_:6.4f}')
        if node.is_expanded:
            args = np.argsort(node.next_N)
            if cutoff: args = args[-cutoff:][::-1]
            children = [ node.next[arg] for arg in args if arg in node.next ]
            for child in children:
                self.print_tree(child, indent+2, cutoff=cutoff)


class TT(object):
    """
    To decide the next move -> to find a =~ pi
    pi(a|s) = N(s,a)^(1/tau) / Sigma_b N(s,b)^(1/tau)
    a_t = argmax_a (Q(s,a) + u(s,a)) 

    u(s,a) = c_puct * P(s,a) * sqrt(Sigma_b N(s,b)) / (1 + N(s,a))
    N(s,a) = Sigma_i^n 1(s,a,i)
    Q(s,a) = W(s,a) / N(s,a), where W(s,a) := W(s,a) + v
    (P(s,-), v) = f_theta(s)

    """
    def __init__(self, net):
        self.net = net
        self.reset_tree()

    def reset_tree(self):
        self.root = Node(None, FakeNode())

    def update_root(self, move):
        if move in self.root.next:
            self.root = self.root.next[move]
            self.root.prev = FakeNode()
        else:
            self.reset_tree()

    def search(self, board):
        """ single search without any MC rollouts
        process (select -> expand and evaluate -> backup) 1x
        """
        turn_to_play = board.whose_turn()
        leaf = self.root.select(board)
        winner = board.check_game_end()
        if winner:
            v = winner == turn_to_play and -1.0 or 1.0
        else:
            P, v = self.fn_policy_value(board)
            leaf.expand(P)
        leaf.backup(-v)

    def fn_pi(self, board, num_search=N_SEARCH):
        """ board -> ( pi(a|s) as [prob] )
        get policy pi as defined in the zero paper
        pi(a|s) = N(s,a)^(1/tau) / Sigma_b N(s,b)^(1/tau)
        tau: temperature controling the degree of exploration 
        simply the normalized visit count when tau=1.
        the smaller tau, the more relying on the visit count.

        """
        for _ in range(num_search): self.search(deepcopy(board))
        tau = board.moves < 5 and 1 or 1e-3
        visits = self.root.next_N
        pi = softmax(1./tau * np.log(np.array(visits)+1) + 1e-10)
        return pi

    def fn_policy_value(self, board):
        """ board -> ( P(s,-), v )
        (p,v) = f_theta(s)
        get policy p and value fn v from network feed-forward.
        p := P(s,-), where P(s,a) = Pr(a|s)
        """
        S = mario.read_state(board)
        S = torch.FloatTensor(S)
        logP, v = self.net(S)
        v = v.flatten().item()
        P = np.exp(logP.flatten().data.numpy())
        return P, v



