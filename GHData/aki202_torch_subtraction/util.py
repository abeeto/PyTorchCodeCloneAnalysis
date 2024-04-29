# -*- coding: utf-8 -*-

def get_char2id():
  char2id = { str(i) : i for i in range(10) }
  char2id.update({ ' ': 10, '-': 11, '_': 12 })
  return char2id
