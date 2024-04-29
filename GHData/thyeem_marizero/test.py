import numpy as np
import torch
import sys
import random
from marizero import MariZero
from board import Board

mario = MariZero()

b = Board()
b.make_move(3,4)
b.make_move(1,1)
b.make_move(3,5)
b.make_move(1,2)
b.make_move(3,6)
b.make_move(3,3)
b.make_move(3,7)

b.make_move(9,9)
b.make_move(5,3)
b.make_move(5,9)
b.make_move(9,1)
b.make_move(5,2)
b.make_move(8,9)
b.make_move(13,7)
b.make_move(1,6)
print(b._)

x, y = mario.next_move(b)
print(x, y)
