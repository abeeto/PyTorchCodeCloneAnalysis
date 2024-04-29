from const import N, Stone, Player, API, BRD_DATA
from sofiai import SofiAI
from mariai import MariAI
from marizero import MariZero
import numpy as np
import os

class Game(object):
    def __init__(self, screen, board):
        self.screen = screen
        self.board = board
        self.pause = False
        self.eB = 50.
        self.eW = 50.
        self.winner = None
        if os.path.exists(BRD_DATA): os.remove(BRD_DATA)

    def get_player(self, stone):
        if stone == Stone.BLACK: return self.playerB
        if stone == Stone.WHITE: return self.playerW

    def set_player(self, playerB, playerW):
        self.playerB = playerB
        self.playerW = playerW

    def nick_player(self, stone):
        player = stone == Stone.BLACK and self.playerB or self.playerW
        return {
            Player.HUMAN: 'Human',
            Player.SOFIAI: 'SofiAI',
            Player.MARIAI: 'MariAI',
            Player.MARIZERO: 'MariZero',
        }.get(player)

    def play_game(self):
        self.screen.init_play_screen(self)
        sofia = SofiAI(self)
        maria = MariAI(self)
        mario = MariZero(self)

        while True:
            if self.winner: break
            x, y = -1, -1
            violate = False
            player = self.get_player(self.board.whose_turn())

            if player == Player.HUMAN:
                x, y = self.screen.get_move()
                if x == -1 or y == -1: return 
                violate = self.board.is_illegal_move(x, y)
                if not violate: self.board.make_move(x, y)

            elif player == Player.SOFIAI:
                x, y = sofia.next_move(API)
                if self.pause: self.screen.pause()
                if not API: self.board.make_move(x, y)

            elif player == Player.MARIAI:
                x, y = maria.next_move(API)
                if self.pause: self.screen.pause()
                if not API: self.board.make_move(x, y)

            elif player == Player.MARIZERO:
                x, y = mario.next_move(self.board)
                self.board.make_move(x, y)
                if self.pause: self.screen.pause()

            else:
                self.screen.wipe_out_msg()
                self.screen.dump_msg("Next move not found. Press q to exit")
                self.screen.wait_for_key('q')
                return 

            if not violate: 
                self.winner = self.board.check_game_end()
                if API: self.write_board(BRD_DATA)
            self.screen.update_screen(self, violate, y, 2*x)

    def read_board(self, file):
        with open(file, 'r') as f: 
            data = [ int(v) for v in f.read().strip().split(':') ]
        self.board.X, self.board.Y = data[0:2]
        self.board.moves, self.board.turn = data[2:4]
        self.board.scoreB, self.board.scoreW = data[4:6]
        self.eB = data[6] / 10.
        self.eW = 100 - self.eB
        self.board._ = np.array(data[8:], dtype=int)
        self.board._.shape = (N, N)
        self.board.last = -1 * self.board.turn

    def write_board(self, file):
        view = self.board._.ravel()
        eB = int(round(self.eB, 1) * 10)
        eW = 1000 - eB
        out = [ self.board.X, self.board.Y, self.board.moves, \
                self.board.turn, self.board.scoreB, self.board.scoreW, eB, eW ]
        out = [ str(v) for v in out+list(view) ]
        with open(file, 'w') as f: f.write(':'.join(out))

    def main_menu(self):
        self.screen.dump_intro()
        return self.screen.read_key()

    def select_menu(self, key):
        black = white = None
        if key == '1':
            black, white = Player.HUMAN, Player.HUMAN
        elif key == '2':
            yes = self.screen.prompt_can_go_first(Player.SOFIAI)
            black, white = yes and (Player.SOFIAI, Player.HUMAN) \
                                or (Player.HUMAN, Player.SOFIAI)
        elif key == '3':
            yes = self.screen.prompt_can_go_first(Player.MARIAI)
            black, white = yes and (Player.MARIAI, Player.HUMAN) \
                                or (Player.HUMAN, Player.MARIAI)
        elif key == '4':
            yes = self.screen.prompt_can_go_first(Player.MARIZERO)
            black, white = yes and (Player.MARIZERO, Player.HUMAN) \
                                or (Player.HUMAN, Player.MARIZERO)
        elif key == '5':
            yes = self.screen.prompt_can_go_first(None)
            self.pause = yes and True or False
            black, white = Player.SOFIAI, Player.MARIAI
        if black and white: self.set_player(black, white)

