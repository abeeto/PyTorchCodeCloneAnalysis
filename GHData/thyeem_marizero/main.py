import curses
from const import Stone, Player
from board import Board
from game import Game
from screen import Screen

try:
    keys = '12345q'
    while True:
        screen = Screen()
        board = Board()
        g = Game(screen, board)
        key = g.main_menu()
        if not key in keys: continue
        if key == 'q': break
        g.select_menu(key)
        g.play_game()
        curses.endwin()

finally:
    curses.endwin()

            
