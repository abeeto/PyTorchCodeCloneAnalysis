


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np
import torch

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

import random,util,math


class TorchAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.dqn = Dqn(109, 4, self.alpha)
        self.index2Action = [Directions.NORTH, Directions.EAST , Directions.SOUTH, Directions.WEST]
        self.last_reward = 0
        self.scores = []
        self.dqn.load()



    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

    
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        new_state = torch.Tensor(self.getInputs(state)).float().unsqueeze(0)
        #new_state = self.getInputs(state)
        action = self.index2Action[self.dqn.select_action(new_state)]
        #print action
        if action in legalActions:
            return action
        else:
            return Directions.STOP

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        last_signal = self.getInputs(state)
        self.dqn.update(reward, last_signal)

    def getInputs(self, state):
        pac = state.getPacmanPosition()
        size = (state.data.layout.width, state.data.layout.height)

        #5x5 Matrix of walls
        lay_walls = state.getWalls()
        walls = self.getMatrix(np.ones(25).reshape(5, 5), lay_walls, pac, size)

        # 5x5 Matrix of food
        lay_food = state.getFood()
        food = self.getMatrix(np.zeros(25).reshape(5, 5), lay_food, pac, size)

        # 5x5 Matrix of ghosts
        pos_ghosts = state.getGhostPositions()
        ghosts = self.getMatrix2(np.zeros(25).reshape(5, 5),pos_ghosts,pac)

        # 5x5 Matrix of power pellets
        pos_capsules = state.getCapsules()
        capsules = self.getMatrix2(np.zeros(25).reshape(5, 5), pos_capsules, pac)


        #1x5 Matrix of scared Time (max 5 ghosts)
        scary = [0,0,0,0,0]
        i = 0
        for agent in state.data.agentStates:
            if  not agent.isPacman and i < 5:
                scary[i] = (float(agent.scaredTimer))
                i += 1

        #1x5 list of legal actions
        legal = self.getLegalActions(state)
        lgllist = []
        for action in self.index2Action:
            val = 1.0 if action in legal else 0.0
            lgllist.append(val)

        lgllist += scary
        matrix = np.concatenate((walls, food, ghosts, capsules))
        return np.append(matrix.flatten(),lgllist)


    def getMatrix2(self,num, list, pac):
        for pos in list:
            if pac[0] - 3 < pos[0] < pac[0] + 3 and pac[1] - 3 < pos[1] < pac[1] + 3:
                x = int(pos[0] - pac[0] + 2)
                y = int(pos[1] - pac[1] + 2)
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

    def final(self, state):
        ReinforcementAgent.final(self, state)
        self.dqn.save()