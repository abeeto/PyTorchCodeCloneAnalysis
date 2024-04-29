import math
import subprocess
from copy import deepcopy
from const import Stone, N, INF, D_MINIMAX, API_SOFIA, BRD_DATA

class SofiAI(object):

    def __init__(self, game):
        self.game = game
        self.board = game.board
        self.coeff = {}
        self.score = {}
        self.prior = {}
        self.d = {} 
        self.init_params()
        
    def init_params(self):
        self.coeff['depth']  = +1.0e2
        self.coeff['score']  = +1.2e4
        #-----------------------------
        self.score['_ooooa'] = -1.0e4;
        self.score['_xxxxa'] = +3.0e6;
        self.score['oxxxxa'] = +1.0e5;
        self.score['oxxa'  ] = -3.0e4;
        self.score['xooa'  ] = +3.0e4;
        self.score['_oooa' ] = -2.0e4;
        self.score['_xxxa' ] = +2.0e4;
        #-----------------------------
        self.prior['oa'   ]  = 0;
        self.prior['xa'   ]  = 0;
        self.prior['ooooa']  = 1;
        self.prior['xxxxa']  = 1;
        self.prior['oooa' ]  = 2;
        self.prior['xxxa' ]  = 2;
        self.prior['ooo_a']  = 3;
        self.prior['xxx_a']  = 3;
        self.prior['oao'  ]  = 4;
        self.prior['xax'  ]  = 4;
        self.prior['_o_oa']  = 5;
        self.prior['_x_xa']  = 5;
        self.prior['_ooa' ]  = 5;
        self.prior['_xxa' ]  = 5;
        self.prior['oxxa' ]  = 6;
        self.prior['xooa' ]  = 6;

    def ask_api(self, file):
        subprocess.run([API_SOFIA, file])

    def next_move(self, api=True): 
        if api: 
            self.ask_api(BRD_DATA)
            self.game.read_board(BRD_DATA)
            return self.board.X, self.board.Y

        else: 
            if self.board.moves == 0: 
                return N//2, N//2
            board = deepcopy(self.board)
            _, x, y = self.minimax(board, -1, -1, 0, True, -INF, +INF)
            return x, y

    def minimax(self, board, x, y, depth, is_maximizer, alpha, beta): 
        bestx = besty = 0
        if x > 0 and y > 0: 
            board.make_move(x, y)
            if board.check_game_end(): 
                if is_maximizer: 
                    bestval = -INF + self.coeff['depth'] * depth
                else:
                    bestval = +INF - self.coeff['depth'] * depth
                return bestval, x, y

            if depth == D_MINIMAX:
                bestval = self.evaluate_state(deepcopy(board), depth)
                return bestval, x, y

        #----------------------------------------------------------------

        child = self.get_child(board)
        if is_maximizer:
            bestval = -INF
            for i, j in child:
                val, _, _ = self.minimax(deepcopy(board), i, j, 
                                         depth+1, False, alpha, beta)
                if depth == 0:
                    if val > bestval:
                        bestval = val 
                        bestx = i
                        besty = j
                else:
                    bestval = max(bestval, val)
                alpha = max(alpha, bestval)
                if beta <= alpha: break
        else:
            bestval = +INF
            for i, j in child:
                val, _, _ = self.minimax(deepcopy(board), i, j, 
                                         depth+1, True, alpha, beta)
                bestval = min(bestval, val)
                beta = min(beta, bestval) 
                if beta <= alpha: break
        return bestval, bestx, besty

    def evaluate_state(self, board, depth): 
        x = board.last_turn()
        o = x == Stone.BLACK and Stone.WHITE or Stone.BLACK

        val_depth = self.coeff['depth'] * depth 
        val_score = self.coeff['score'] * \
                    (math.pow(board.get_score(o), 1.4) - 
                     math.pow(board.get_score(x), 1.4))
        val_pt = 0
        self.get_pt_score(board, o, x)
        for pt, e in self.d.items():
            if not pt in self.score: continue
            val_pt += self.score[pt] * len(e)
        return val_depth + val_score + val_pt

    def get_child(self, board): 
        x = board.last_turn()
        o = x == Stone.BLACK and Stone.WHITE or Stone.BLACK
        self.get_pt_candidate(board, o, x)
        if not len(self.d): self.get_pt_no_child(board, o, x)
        candy = sorted(self.d.items(), 
                       key=lambda x: self.prior[x[0]], reverse=True)
        child = [ e for _, es in candy for e in es ]
        return child

    def char_to_stone(self, char, o, x): 
        if char == 'o':
            return o
        elif char == 'x': 
            return x
        else:
            return Stone.EMPTY

    def get_pt_score(self, board, o, x): 
        self.d.clear()
        for i in range(N):
            for j in range(N):
                if board.get_stone(i, j) == Stone.EMPTY:
                    self.find_pt(board, i, j, o, x, '_oooa')
                    self.find_pt(board, i, j, o, x, '_ooooa')
                    self.find_pt(board, i, j, o, x, '_xxxa')
                    self.find_pt(board, i, j, o, x, '_xxxxa')
                else:
                    self.find_pt(board, i, j, o, x, 'oxxa')
                    self.find_pt(board, i, j, o, x, 'xooooa')
                    self.find_pt(board, i, j, o, x, 'xooa')

    def get_pt_candidate(self, board, o, x): 
        self.d.clear()
        for i in range(N):
            for j in range(N):
                if board.get_stone(i, j) == Stone.EMPTY: 
                    self.find_pt(board, i, j, o, x, '_ooa')
                    self.find_pt(board, i, j, o, x, '_xxa')
                    self.find_pt(board, i, j, o, x, '_x_xa')
                    self.find_pt(board, i, j, o, x, '_o_oa')
                else:
                    self.find_pt(board, i, j, o, x, 'oooa')
                    self.find_pt(board, i, j, o, x, 'xxxa')
                    self.find_pt(board, i, j, o, x, 'oxxa')
                    self.find_pt(board, i, j, o, x, 'xooa')
                    self.find_pt(board, i, j, o, x, 'oao')
                    self.find_pt(board, i, j, o, x, 'xax')
                    self.find_pt(board, i, j, o, x, 'ooo_a')
                    self.find_pt(board, i, j, o, x, 'xxx_a')

    def get_pt_no_child(self, board, o, x): 
        self.d.clear()
        for i in range(N):
            for j in range(N):
                if board.get_stone(i, j) == Stone.EMPTY: continue
                else:
                    self.find_pt(board, i, j, o, x, 'oa')
                    self.find_pt(board, i, j, o, x, 'xa')

    def find_pt(self, board, i, j, o, x, pt):
        self.find_pt_along(board, i, j,  1,  0, o, x, pt)
        self.find_pt_along(board, i, j, -1,  0, o, x, pt)
        self.find_pt_along(board, i, j,  0,  1, o, x, pt)
        self.find_pt_along(board, i, j,  0, -1, o, x, pt)
        self.find_pt_along(board, i, j,  1,  1, o, x, pt)
        self.find_pt_along(board, i, j,  1, -1, o, x, pt)
        self.find_pt_along(board, i, j, -1,  1, o, x, pt)
        self.find_pt_along(board, i, j, -1, -1, o, x, pt)

    def find_pt_along(self, board, i, j, di, dj, o, x, pt): 
        for s in range(len(pt)):
            if not board.is_inside(i+s*di, j+s*dj): return 
            if board.get_stone(i+s*di, j+s*dj) != \
               self.char_to_stone(pt[s], o, x): return 

        for s in range(len(pt)):
            if pt[s] == 'a': 
                if pt in self.d: self.d[pt].add((i+s*di, j+s*dj,))
                else: self.d[pt] = {(i+s*di, j+s*dj,)}

