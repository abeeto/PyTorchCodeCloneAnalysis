import curses
from const import Stone, Player, N
class Screen(object):

    def __init__(self):
        self.s = curses.initscr()
        self.s.keypad(True)
        self.s.refresh()
        curses.cbreak()
        curses.noecho()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED,    curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE,  curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_BLUE,   curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_GREEN,  curses.COLOR_BLACK)

    def init_play_screen(self, game):
        self.s.clear()
        self.s.refresh()
        self.w = curses.newwin(N+2, 2*N+3, (curses.LINES-N)//2, (curses.COLS-2*N)//2)
        self.w.box(0, 0)
        self.w.refresh()
        self.w = self.w.derwin(N, 2*N, 1, 2)
        self.w.keypad(True)
        self.update_screen(game, 9, N//2, N-1)

    def read_key(self, sub=False, char=True):
        win = sub and self.w or self.s
        key = char and chr(win.getch()) or win.getch()
        return key

    def dump_intro(self):
        self.s.clear()
        self.s.addstr(curses.LINES//2-8, curses.COLS//2-20, "BCF AIs WELCOMES YOU.");
        self.s.addstr(curses.LINES//2-6, curses.COLS//2-20, "[1] PLAYING WITH HUMAN");
        self.s.addstr(curses.LINES//2-4, curses.COLS//2-20, "[2] PLAYING WITH SOFIAI");
        self.s.addstr(curses.LINES//2-2, curses.COLS//2-20, "[3] PLAYING WITH MARIAI");
        self.s.addstr(curses.LINES//2+0, curses.COLS//2-20, "[4] PLAYING WITH MARIZERO");
        self.s.addstr(curses.LINES//2+2, curses.COLS//2-20, "[5] WATCHING A GAME BETWEEN AIs");
        self.s.addstr(curses.LINES//2+4, curses.COLS//2-20, "[q] EXIT");
        self.s.addstr(curses.LINES//2+6, curses.COLS//2-20, "PRESS THE KEY TO CONTINUE");

    def get_confirm(self, prompt):
        self.s.addstr(curses.LINES//2+6, curses.COLS//2-20, prompt)
        return self.read_key()

    def prompt_can_go_first(self, player):
        ans = None
        if player == Player.SOFIAI:
            ans = self.get_confirm("DO YOU ALLOW SOFIAI TO PLAY FIRST? [y/N]: ")
        elif player == Player.MARIAI:
            ans = self.get_confirm("DO YOU ALLOW MARIAI TO PLAY FIRST? [y/N]: ")
        elif player == Player.MARIZERO:
            ans = self.get_confirm("DO YOU ALLOW MARIZERO TO PLAY FIRST? [y/N]: ")
        elif not player:
            ans = self.get_confirm("DO YOU LET AIs PAUSE BETWEEN EACH THEIR MOVE? [y/N]: ")
        if ans: return ans in 'yY'

    def wipe_out_msg(self):
        self.dump_msg(50*' ')
        self.s.refresh()

    def wipe_out_progress(self):
        self.s.addstr((curses.LINES-N)//2+21, (curses.COLS-2*N)//2, 50*' ')
        self.s.refresh()

    def dump_msg(self, msg):
        self.s.addstr((curses.LINES-N)//2-1, (curses.COLS-2*N)//2, f'Message: {msg}')
        self.s.refresh()

    def dump_turn(self, board):
        self.s.move((curses.LINES-N)//2-3, (curses.COLS-2*N)//2)
        self.s.addstr("Whose Turn: ")
        if board.whose_turn() == Stone.BLACK:
            self.s.attron(curses.color_pair(3))
            self.s.addstr("Black")
            self.s.attroff(curses.color_pair(3))
        else:
            self.s.attron(curses.color_pair(1))
            self.s.addstr("White")
            self.s.attroff(curses.color_pair(1))
        self.s.addstr(f'\tMoves: {board.moves}')

    def dump_score(self, game):
        self.s.move((curses.LINES-N)//2-2, (curses.COLS-2*N)//2)
        self.s.attron(curses.color_pair(3))
        self.s.addstr("Black")
        self.s.attroff(curses.color_pair(3))
        self.s.addstr(f' [{game.nick_player(Stone.BLACK)}]:' +
                      f' {game.board.get_score(Stone.BLACK)}')
        self.s.addstr('\t')
        self.s.attron(curses.color_pair(1))
        self.s.addstr("White")
        self.s.attroff(curses.color_pair(1))
        self.s.addstr(f' [{game.nick_player(Stone.WHITE)}]:' +
                      f' {game.board.get_score(Stone.WHITE)}')

    def dump_who_won(self, winner):
        if not winner: return False
        winner = winner == Stone.BLACK and "BLACK" or "WHITE"
        self.dump_msg(f'{winner} won. Press q to exit')
        self.wait_for_key('q')
        return True

    def dump_EWP(self, eB, eW):
        self.s.addstr((curses.LINES-N)//2-1, (curses.COLS-2*N)//2,
                      f'Message: [EWP] B({eB:.1f}%) - W({eW:.1f}%)')

    def wait_for_key(self, key=None):
        if not key and self.read_key(): return 
        while True:
            if self.read_key() == key: return 

    def pause(self):
        self.dump_msg("Press any key to continue")
        self.wait_for_key()

    def update_screen(self, game, code, y, x):
        """ violation code: 
        0  legal 
        1  overlap
        2  3-3 illegal 
        9  new game
        """
        self.wipe_out_msg()
        self.wipe_out_progress()
        if code == 0: self.dump_EWP(game.eB, game.eW)
        elif code == 1: self.dump_msg("Oops! Are you serious?")
        elif code == 2: self.dump_msg("Thas's an illegal move: 3-3 connected.")
        elif code == 9: self.dump_msg("Get started a new game.")
        self.dump_board(game.board)
        self.dump_turn(game.board)
        self.dump_score(game)
        if self.dump_who_won(game.winner): return
        self.s.refresh()
        self.w.move(y, x)
        self.w.refresh()

    def dump_board(self, board):
        for i in range(N):
            for j in range(N):
                self.w.attron(curses.color_pair(board.get_stone(i,j)+2))
                if board.get_stone(i, j) == Stone.EMPTY:
                    self.w.addch(j, 2*i, curses.ACS_BULLET)
                elif board.get_stone(i, j) == Stone.BLACK:
                    self.w.addch(j, 2*i, 'x')
                elif board.get_stone(i, j) == Stone.WHITE:
                    self.w.addch(j, 2*i, 'o')
                self.w.attroff(curses.color_pair(board.get_stone(i,j)+2))

                if board.is_last_move(i, j) and board.moves > 0:
                    self.w.attron(curses.color_pair(5))
                    if board.get_stone(i, j) == Stone.BLACK: self.w.addch(j, 2*i, 'x')
                    else: self.w.addch(j, 2*i, 'o')
                    self.w.attroff(curses.color_pair(5))
        self.w.refresh()

    def get_move(self):
        while True:
            key = self.read_key(True, False)
            if key == ord('\n'): break
            if key == ord('q'): return -1, -1

            h, w = self.w.getmaxyx()
            y, x = self.w.getyx()
            if key == curses.KEY_UP:
                if y > 0: self.w.move(y-1, x)
            elif key == curses.KEY_DOWN:
                if y < N-1: self.w.move(y+1, x)
            elif key == curses.KEY_LEFT:
                if x > 0: self.w.move(y, x-2)
            elif key == curses.KEY_RIGHT:
                if x < 2*(N-1): self.w.move(y, x+2)
        y, x = self.w.getyx()
        return x//2, y


