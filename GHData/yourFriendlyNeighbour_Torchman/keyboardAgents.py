# keyboardAgents.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Agent
from game import Directions
import random
import numpy as np

class KeyboardAgent(Agent):
    """
    An agent controlled by the keyboard.
    """
    # NOTE: Arrow keys also work.
    WEST_KEY  = 'a'
    EAST_KEY  = 'd'
    NORTH_KEY = 'w'
    SOUTH_KEY = 's'
    STOP_KEY = 'q'

    def __init__( self, index = 0 ):

        self.lastMove = Directions.STOP
        self.index = index
        self.keys = []

    def getAction( self, state):
        from graphicsUtils import keys_waiting
        from graphicsUtils import keys_pressed
        keys = keys_waiting() + keys_pressed()
        if keys != []:
            self.keys = keys

        legal = state.getLegalActions(self.index)
        move = self.getMove(legal)

        if move == Directions.STOP:
            # Try to move in the same direction as before
            if self.lastMove in legal:
                move = self.lastMove

        if (self.STOP_KEY in self.keys) and Directions.STOP in legal: move = Directions.STOP

        if move not in legal:
            move = random.choice(legal)

        self.lastMove = move
        return move

    def getMove(self, legal):
        move = Directions.STOP
        if   (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:  move = Directions.WEST
        if   (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal: move = Directions.EAST
        if   (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:   move = Directions.NORTH
        if   (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

    def final(self, state):
        self.getInputs(state)
       # layout = state.data.layout
        #print layout.numGhosts #Anzahl Geister als Integer
       # for item in layout.layoutText:
        #    print item




    def getInputs(self, state):
        pac = state.getPacmanPosition()
        size = (state.data.layout.width, state.data.layout.height)

        lay_walls = state.getWalls()
        walls = self.getMatrix(np.ones(25).reshape(5, 5), lay_walls, pac, size)

        lay_food = state.getFood()
        food = self.getMatrix(np.zeros(25).reshape(5, 5), lay_food, pac, size)

        pos_ghosts = state.getGhostPositions()
        ghosts = self.getMatrix2(np.zeros(25).reshape(5, 5),pos_ghosts,pac)

        pos_capsules = state.getCapsules()
        capsules = self.getMatrix2(np.zeros(25).reshape(5, 5), pos_capsules, pac)


    def getMatrix2(self,num, list, pac):
        for pos in list:
            if pac[0] - 3 < pos[0] < pac[0] + 3 and pac[1] - 3 < pos[1] < pac[1] + 3:
                x = int(pos[0] - pac[0] + 2)
                y = int(pos[1] - pac[1] + 2)
                print (x, y, pac[0], pos[0], pac[1], pos[1])
                num[x, y] = 1
        return num


    def getMatrix(self, num, lay, pac, size):
        width = size[0]
        height = size[1]

        for x in xrange(5):
            for y in xrange(5):
                new_x = pac[0] - 2 + x
                new_y = pac[1] - 2 + y
                if (-1 < new_x < width) and (-1 < new_y < height):
                    if lay[new_x][new_y]:
                        num[x, y] = 1
                    else:
                        num[x,y] = 0
        return num


class KeyboardAgent2(KeyboardAgent):
    """
    A second agent controlled by the keyboard.
    """
    # NOTE: Arrow keys also work.
    WEST_KEY  = 'j'
    EAST_KEY  = "l"
    NORTH_KEY = 'i'
    SOUTH_KEY = 'k'
    STOP_KEY = 'u'

    def getMove(self, legal):
        move = Directions.STOP
        if   (self.WEST_KEY in self.keys) and Directions.WEST in legal:  move = Directions.WEST
        if   (self.EAST_KEY in self.keys) and Directions.EAST in legal: move = Directions.EAST
        if   (self.NORTH_KEY in self.keys) and Directions.NORTH in legal:   move = Directions.NORTH
        if   (self.SOUTH_KEY in self.keys) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
