import os
import chess.pgn
import numpy as np
from state import State

def test_dataset(num_samples=None):
  X,Y = [], []
  gn = 0
  values = {'1/2-1/2':0, '0-1':-1, '1-0':1}
    pgn = open(os.path.join("data", fn))
  while 1:
    try:
      game = chess.pgn.read_game(pgn)
    except:
      break

