import numpy as np
import gym
from gym import spaces

class RepeatedBoSEnv(gym.Env):
    def __init__(self, partner_policies, horizon):
        super(RepeatedBoSEnv, self).__init__()
        """Implementation of a repeated simultaneous game (Bach or Stravinsky)
        
        partner_policy: function that determines the policy of the partner agent
        horizon: number of games to play
        """
        self.partner_policies = partner_policies
        self.horizon = horizon
        self.action_space = spaces.Discrete(2)
        
        # state is the actions of both agents in the previous game
        # initialize state randomly since there was no previous game
        self.state_space = spaces.MultiDiscrete([2, 2]) 
        self.state = self.state_space.sample()

        # state and action history (action includes only human action)
        self.prev_state = self.state_space.sample()
        self.prev_h_action = self.action_space.sample()

        # observation is the previous state, and a 1-step (state, action) pair history
        self.observation_space = spaces.MultiDiscrete([2, 2, 2, 2, 2]) # includes state and action

        self.game_num = 0

    def step(self, action):

        self.game_num += 1

        partner_action = self.partner_policy(self.state)

        # set previous state and action
        self.prev_state = self.state
        self.prev_h_action = partner_action

        if action == 0 and partner_action == 0:
            # both agents decide on Bach
            reward = 3
        elif action == 0 and partner_action == 1:
            # ego agent decides on Bach and partner on Stravinsky
            reward = 1
        elif action == 1 and partner_action == 0:
            # ego agent decides on Stravinsky and partner on Bach
            reward = 0
        elif action == 1 and partner_action == 1:
            # both agents decide on Stravinsky
            reward = 2

        joint_action = (action, partner_action)
        self.state = joint_action

        obs = np.hstack((self.state, self.prev_state, self.prev_h_action))

        return obs, reward, (self.game_num >= self.horizon), {}

    def reset(self):
        # choose one partner policy to use for this rollout
        self.partner_policy = self.partner_policies[np.random.randint(len(self.partner_policies))]
        self.game_num = 0
        observation = self.observation_space.sample()
        # self.state = observation
        self.state = self.state_space.sample()
        self.prev_state = self.state_space.sample()
        self.prev_h_action = self.action_space.sample()

        return observation


