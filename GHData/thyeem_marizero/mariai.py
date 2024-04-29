import subprocess
from const import API_MARIA, BRD_DATA

class MariAI(object):
    def __init__(self, game):
        self.game = game
        self.board = game.board
        self.screen = game.screen

    def ask_api(self, file):
        subprocess.run([API_MARIA, file])

    def next_move(self, api=True): 
        if api: 
            self.ask_api(BRD_DATA)
            self.game.read_board(BRD_DATA)
            return self.board.X, self.board.Y

