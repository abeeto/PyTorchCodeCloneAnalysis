import gym
from gym.spaces import Box
from gym.envs.registration import EnvSpec
from gym.envs.registration import register

import numpy as np
from numpy.linalg import norm
from numpy.random import random

from PointEnvironment.Agent import Agent
from PointEnvironment.Pose import Pose


class Go2Goal(gym.Env):

    cossin = staticmethod(lambda x: np.array([np.cos(x), np.sin(x)]))

    def __init__(self, config=None):
        """
        Simple go to goal environment where a non-holonomic agent is
        required to move to a goal. The goal can be random or single,
        however it is advised that during training we provide it a
        random goal everytime. Rewards are binary.

        config includes:
        max_episode_steps (int): maximum number of timesteps in an episode
        reward_max (int): reward when goal is achieved.
        seed (int): seed of the random numpy process.
        her (bool): whether to use the HER compatible variant or not
        dt (float): dt in kinematic update equation
        num_iter (int): num of iterations of kinematic update equation

        """
        # Default values. Will be overridden if specified in config
        self.dt = 1e-2
        # self.her = True
        self.her = True
        # self.seed = None
        self.thresh = np.array([0.05, 0.05, 0.1])[:-1]
        self.num_iter = 50
        self.reward_max = 1
        self.max_episode_steps = 25
        self._max_episode_steps = 25
        self.step_penalty = 1.0  # (self.max_episode_steps)

        self.action_low = np.array([0.0, -np.pi/4])
        self.action_high = np.array([0.3, np.pi/4])
        self.action_space = Box(self.action_low, self.action_high, dtype="f")

        # so that the goals are within the range of performing actions
        self.d_clip = self.action_high[0]*self.num_iter*self.dt*1.35

        self.observation_space = Box(low=-1, high=1, shape=(5,), dtype="f")

        self.limits = np.array([1, 1, np.pi])
        self.agent = Agent(0)
        if config is not None:
            self.__dict__.update(config)

        # if self.seed is not None:
        #     np.random.seed(self.seed)

        self.goal = None
        if not self.her:
            self.dMax = self.action_high[0]*self.dt*self.num_iter
            self.dRange = 2*self.dMax
        self.viewer = None
        self._spec = EnvSpec("Go2Goal-v0")

    def reset(self):
        self.agent.reset(self.sample_pose())
        self.goal = self.sample_pose()

        # print("Agent spawned at: ", self.agent.pose)
        # print("Goalpoint set to: ", self.goal)
        return self.compute_obs()

    def step(self, action):
        assert self.goal is not None, "Call reset before calling step"
        prev_pose = self.agent.pose.clone()
        for i in range(self.num_iter):
            self.agent.step(action, dt=self.dt)
        reward, done = self.get_reward()
        return self.compute_obs(prev_pose), reward, done,\
            {"dist": self.current_distance, "is_success": done}

    def get_reward(self):
        reached = (self.distance_from_goal() < self.thresh).all()
        if reached:
            return self.reward_max, True
        return -self.step_penalty, False
        # return -self.step_penalty if not reached else self.reward_max
        # reward = self.reward_max if reached else 0.0
        # return reward-self.step_penalty, reached

    def distance_from_goal(self):
        self.current_distance = np.abs((self.agent.pose - self.goal))[:-1]
        return self.current_distance

    def sample_pose(self):
        return Pose(*(random(3)*self.limits - self.limits/2))

    def render(self, mode='human', extra=None):
        screen_width = 500
        screen_height = 500

        world_width = 2.5
        scale = screen_width/world_width
        if self.goal is None:
            return None
        c, theta = np.split(self.goal.tolist(), [2])
        a, atheta = np.split(self.agent.pose.tolist(), [2])
        sh = 20
        ss = 5
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # Draw grids
            # Todo: write better code for making grids...
            for i in np.arange(-1.5, 1.5, step=0.5):
                pt = i*scale + screen_width/2.
                line = rendering.Line((0, pt), (screen_width, pt))
                line.set_color(0.5, 0.55, 0.52)
                self.viewer.add_geom(line)
                pt = i*scale + screen_width/2.
                line = rendering.Line((pt, 0), (pt, screen_width))
                line.set_color(0.5, 0.55, 0.52)
                self.viewer.add_geom(line)

            self.goal_vec = rendering.Line((0, 0), (10, 10))
            self.goal_vec.set_color(1., 0.01, 0.02)
            self.viewer.add_geom(self.goal_vec)

            goal = rendering.make_circle()
            goal.set_color(0.3, 0.82, 0.215)
            self.visual_goal_trans = rendering.Transform()
            goal.add_attr(self.visual_goal_trans)
            self.viewer.add_geom(goal)

            vs = [c+Go2Goal.cossin(0)*sh, c+Go2Goal.cossin(-np.pi/2)*ss,
                  c+Go2Goal.cossin(np.pi/2)*ss]
            agent = rendering.FilledPolygon([tuple(i) for i in vs])
            agent.set_color(0.15, 0.235, 0.459)
            self.visual_agent_trans = rendering.Transform()
            agent.add_attr(self.visual_agent_trans)
            self.viewer.add_geom(agent)
            self.traj_geom = []
            if extra is not None:
                count = 0
                for start, end in zip(extra[:-1], extra[1:]):
                    start, end = start*scale+screen_width/2.0, end*scale+screen_width/2.0
                    self.traj_geom.append(rendering.Line(start, end))
                    # self.traj_geom[-1].set_color(count/len(extra), 1-(count/len(extra)), 0.)
                    self.traj_geom[-1].set_color(count/len(extra), 1-(count/len(extra)), 0.)
                    self.viewer.add_geom(self.traj_geom[-1])
                    count+=1

        self.goal_vec.start = tuple((a*scale+screen_width/2.0))
        self.goal_vec.end = tuple((a*scale+screen_width/2.0) +
                                  self.obs["desired_goal"][:-1] *
                                  self.obs["desired_goal"][-1] * scale)

        self.visual_goal_trans.set_translation(*(c*scale+screen_width/2.0))
        self.visual_goal_trans.set_rotation(theta)
        self.visual_agent_trans.set_translation(*(a*scale+screen_width/2.0))
        self.visual_agent_trans.set_rotation(atheta)
        if extra is not None:
            for start, end, line in zip(extra[:-1], extra[1:], self.traj_geom):
                line.start, line.end = start*scale+screen_width/2.0, end*scale+screen_width/2.0

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def compute_obs(self, prev_pose=None):
        # our observations are going to be the distance to be moved in
        # the direction of the goal in the vector form...
        # to be brief obs =  pos vec of goal relative to agent...
        # but what about angles? lets ignore them for now...
        goal_vec, angle = np.split((self.goal - self.agent.pose), [-1])
        distance = np.linalg.norm(goal_vec)
        unit_vec = goal_vec / distance if distance != 0 else goal_vec
        # c_dist = min(distance, self.d_clip)#/self.d_clip
        c_dist = distance
        sc = np.hstack([np.cos(self.agent.pose.theta),
                        np.sin(self.agent.pose.theta)])
        goal = np.hstack([unit_vec, c_dist])
        if not self.her:
            return np.hstack([sc, goal])
        if prev_pose is None:
            ag = np.zeros(3)
        else:
            ag = (self.agent.pose - prev_pose)[:-1]
            if norm(ag) < 1e-6:
                ag += 1e-8

            ag = np.hstack([ag/norm(ag), norm(ag)])
        self.obs = {"observation": sc,
                    "desired_goal": goal,
                    "achieved_goal": ag}
        return self.obs

    def close(self):
            if self.viewer:
                self.viewer.close()
                self.viewer = None

    def seed(self, seed):
        np.random.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # print(achieved_goal.shape)
        # print(desired_goal.shape)
        r = norm(achieved_goal - desired_goal, axis=1)
        # print(r, r.shape, r.astype('i'))
        return (r < self.thresh).astype('i')


register(
    id='Go2Goal-v0',
    entry_point='go2goal:Go2Goal',
)
