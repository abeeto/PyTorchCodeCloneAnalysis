import chess
import time
import torch
import chess.svg
import traceback
import base64
from state import State
from flask import Flask, response, request
from train import Net


class Valuator(object):
  def __init__(self):
    vals = torch.load("nets/value.pth", map_location=lambda storage, loc: storage)
    self.model = Net()
    self.model.load_state_dict(vals)

  def __call__(self, s):
    brd = s.serialize()[None]
    output = self.model(torch.tensor(brd).float())
    return float(output.data[0][0])


class Handler:
  def __init__(self):
    return None
    

  app = Flask(__name__)
  s = State()
  v = Valuator()
  
  def explore_leaves(self, s, v):
    ret = []
    for e in s.edges():
      self.s.board.push(e)
      ret.append((v(s), e))
      self.s.board.pop()
    return ret

  def to_svg(self, s):
    return base64.b64encode(chess.svg.board(board=self.s.board).encode('utf-8')).decode('utf-8')

  def hello(self):
    board_svg = self.to_svg(self.s)
    ret = '<html><head>'
    ret += '<style>input { font-size: 30px; } button { font-size: 30px; }</style>'
    ret += '</head><body>'
    ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % self.board_svg
    ret += '<form action="/move"><input name="move" type="text"></input><input type="submit" value="Move"></form><br/>'
    return ret
  
  def computer_move(self, s, v):
    move = sorted(self.explore_leaves(self.s, self.v), key=lambda x: x[0], reverse=self.s.board.turn)
    print("top 3:")
    for i,m in enumerate(self.move[0:3]):
      print("  ",m)
    self.s.board.push(self.move[0][1])

  def selfplay(self):
    s = State()
    ret = '<html><head>'
    while not self.s.board.is_game_over():
      self.computer_move(self.s, self.v)
      ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % self.to_svg(s)
    print(self.s.board.result())
    return ret

  def move(self):
    if not self.s.board.is_game_over():
      move = request.args.get('move',default="")
      if move is not None and move != "":
        print("human moves", move)
        try:
          self.s.board.push_san(move)
          self.computer_move(self.s, self.v)
        except Exception:
          print "Error"
    else:
      print("Gover over")
    return self.hello()


if __name__ == '__main__':
  handler = Handler()
  handler.app.add_app_rule('/', 'index', handler.hello)
  handler.app.add_app_rule('/selfplay', 'selfplay', handler.selfplay)
  handler.app.add_app_rule('/move', 'makemove', handler.move)
  handler.app.run(debug=True, port=8000)  