#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import WorldDemo
import threading

action = []
actions = WorldDemo.actions

def try_move(action):
    if action == actions[0]:
        WorldDemo.try_move(0, -1)
    elif action == actions[1]:
        WorldDemo.try_move(0, 1)
    elif action == actions[2]:
        WorldDemo.try_move(-1, 0)
    elif action == actions[3]:
        WorldDemo.try_move(1, 0)
    else:
        return

def main():
    print("START")
    print(actions)

t = threading.Thread(target=main)
t.daemon = True
t.start()
WorldDemo.start_game()
