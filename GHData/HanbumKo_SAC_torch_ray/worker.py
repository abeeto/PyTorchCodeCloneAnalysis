import ray
import gym
import numpy as np

from core import *
from copy import deepcopy

@ray.remote
class RayRolloutWorker(object):
    """
    Rollout Worker with RAY
    """
    def __init__(self, envname, hidden=256, l=2, worker_id=0, ep_len_rollout=1000, max_ep_len_rollout=1000):
        print("Maing new rollout worker", worker_id)
        self.worker_id = worker_id
        self.ep_len_rollout = ep_len_rollout
        self.max_ep_len_rollout = max_ep_len_rollout
        # gym.logger.set_level(40)
        self.env = gym.make(envname)
        self.odim = self.env.observation_space.shape
        self.adim = self.env.action_space.shape[0]
        self.o = self.env.reset()

        # Replay buffers to pass
        self.o_buf = np.zeros(combined_shape(self.ep_len_rollout, self.odim), dtype=np.float32)
        self.a_buf = np.zeros(combined_shape(self.ep_len_rollout, self.adim), dtype=np.float32)
        self.r_buf = np.zeros(self.ep_len_rollout, dtype=np.float32)
        self.o2_buf = np.zeros(combined_shape(self.ep_len_rollout, self.odim), dtype=np.float32)
        self.d_buf = np.zeros(self.ep_len_rollout, dtype=np.float32)

        # Create SAC model
        ac_kwargs = dict(hidden_sizes=[hidden]*l)
        self.ac = MLPActorCritic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac) # Is target needed for worker?

        print("Done new worker", worker_id)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def set_weights(self, new_ac_state, new_ac_targ_state):
        self.ac.load_state_dict(new_ac_state)
        self.ac_targ.load_state_dict(new_ac_targ_state)
    
    def rollout(self, eps_greedy=0.0):
        for t in range(self.ep_len_rollout):
            if np.random.rand() < eps_greedy:
                self.a = self.env.action_space.sample()
            else:
                self.a = self.get_action(self.o, deterministic=False)
            self.o2, self.r, self.d, _ = self.env.step(self.a)
            
            # Append
            self.o_buf[t, :] = self.o2
            self.a_buf[t, :] = self.a
            self.r_buf[t] = self.r
            self.o2_buf[t, :] = self.o2
            self.d_buf[t] = self.d

            # Save next state
            self.o = self.o2
            if self.d:
                self.o = self.env.reset()

        return self.o_buf, self.a_buf, self.r_buf, self.o2_buf, self.d_buf


        


