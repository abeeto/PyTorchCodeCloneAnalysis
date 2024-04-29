from const import Stone, N, WP, BCF, GOAL
import numpy as np

class Board(object):

    def __init__(self):
        self.turn = Stone.BLACK
        self.last = Stone.WHITE
        self.scoreB = 0
        self.scoreW = 0
        self.X = -1
        self.Y = -1
        self.moves = 0
        self._ = np.zeros((N, N), dtype=int)

    def get_board(self):
        return self._

    def get_stone(self, x, y):
        return self._[x][y]

    def set_stone(self, x, y, stone):
        self._[x][y] = stone

    def get_score(self, stone):
        if stone == Stone.BLACK: return self.scoreB
        if stone == Stone.WHITE: return self.scoreW

    def set_score(self, stone, score):
        if stone == Stone.BLACK: self.scoreB = score
        if stone == Stone.WHITE: self.scoreW = score

    def is_last_move(self, x, y):
        return x == self.X and y == self.Y

    def get_last_move(self):
        return self.X, self.Y

    def set_last_move(self, x, y):
        self.X = x
        self.Y = y

    def toggle_turn(self):
        self.last = self.turn
        self.turn *= -1

    def whose_turn(self):
        return self.turn 

    def last_turn(self):
        return self.last

    def is_inside(self, x, y):
        if x < 0 or x >= N: return False
        if y < 0 or y >= N: return False
        return True

    def is_illegal_move(self, x, y):
        if self._[x][y] != Stone.EMPTY: return 1 
        if self.check_3_3(x, y): return 2
        return 0

    def make_move(self, x, y, sure=False): 
        if not sure and self.is_illegal_move(x, y): return False
        self.moves += 1
        self.set_stone(x, y, self.turn)
        self.set_last_move(x, y)
        if BCF: self.bite_move(x, y)
        self.toggle_turn()
        return True

    def who_won(self):
        """ should be called in check_game_end fn only
        """
        if self.scoreB >= WP: return Stone.BLACK
        if self.scoreW >= WP: return Stone.WHITE
        return self.last

    def check_game_end(self):
        x, y = self.get_last_move()
        if self.scoreB >= WP or self.scoreW >= WP:
            return self.who_won()
        if self.check_game_end_along(x, y, -1,  0) + \
           self.check_game_end_along(x, y,  1,  0) + 1 == GOAL: 
            return self.who_won()
        if self.check_game_end_along(x, y,  0, -1) + \
           self.check_game_end_along(x, y,  0,  1) + 1 == GOAL: 
            return self.who_won()
        if self.check_game_end_along(x, y, -1, -1) + \
           self.check_game_end_along(x, y,  1,  1) + 1 == GOAL: 
            return self.who_won()
        if self.check_game_end_along(x, y, -1,  1) + \
           self.check_game_end_along(x, y,  1, -1) + 1 == GOAL: 
            return self.who_won()
        return False

    def check_game_end_along(self, x, y, dx, dy):
        rows = 0
        while True:
            x += dx
            y += dy 
            if not self.is_inside(x, y): return rows 
            if self._[x][y] != self.last: return rows 
            else: rows += 1

    def check_3_3(self, x, y):
        count = 0
        if self.check_3_3_along(x, y, -1,  0) + \
           self.check_3_3_along(x, y,  1,  0) + 1 == 3: count += 1
        if self.check_3_3_along(x, y,  0, -1) + \
           self.check_3_3_along(x, y,  0,  1) + 1 == 3: count += 1
        if self.check_3_3_along(x, y, -1, -1) + \
           self.check_3_3_along(x, y,  1,  1) + 1 == 3: count += 1
        if not count: return False
        if self.check_3_3_along(x, y, -1,  1) + \
           self.check_3_3_along(x, y,  1, -1) + 1 == 3: count += 1
        return count >= 2 and True or False

    def check_3_3_along(self, x, y, dx, dy):
        rows = 0
        while True:
            x += dx
            y += dy 
            if not self.is_inside(x, y): return -N
            if self._[x][y] == Stone.EMPTY: return rows 
            if self._[x][y] != self.turn: return -N 
            rows += 1

    def bite_move(self, x, y):
        self.bite_move_along(x, y, -1,  0)
        self.bite_move_along(x, y,  1,  0)
        self.bite_move_along(x, y,  0, -1)
        self.bite_move_along(x, y,  0,  1)
        self.bite_move_along(x, y,  1,  1)
        self.bite_move_along(x, y, -1,  1)
        self.bite_move_along(x, y,  1, -1)
        self.bite_move_along(x, y, -1, -1)

    def bite_move_along(self, x, y, dx, dy):
        if not self.is_inside(x + 3*dx, y + 3*dy): return 
        if self._[x + 3*dx][y + 3*dy] == self.turn and \
           self._[x + 2*dx][y + 2*dy] == self.last and \
           self._[x + 1*dx][y + 1*dy] == self.last:
            self._[x + 2*dx][y + 2*dy] = Stone.EMPTY
            self._[x + 1*dx][y + 1*dy] = Stone.EMPTY
            if self.turn == Stone.BLACK: self.scoreB += 2
            else: self.scoreW += 2

