import itertools
import torch
import gym
import time
import numpy as np

from config import *
from core import *
from worker import *
from replaybuffer import *
from copy import deepcopy
from torch.optim import Adam


env_fn = lambda: gym.make(env)
actor_critic = MLPActorCritic
ac_kwargs = dict(hidden_sizes=[hid]*l)

ray.init(num_cpus=n_cpu)
workers = [RayRolloutWorker.remote(envname=env, hidden=hid, l=l, worker_id=i,
                     ep_len_rollout=max_ep_len, max_ep_len_rollout=max_ep_len)
                     for i in range(n_workers)]

torch.manual_seed(seed)
np.random.seed(seed)

env, test_env = env_fn(), env_fn()
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]

# Action limit for clamping: critically, assumes all dimensions share the same bound!
act_limit = env.action_space.high[0]

# Create actor-critic module and target networks
ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
ac_targ = deepcopy(ac)

# Freeze target networks with respect to optimizers (only update via polyak averaging)
for p in ac_targ.parameters():
    p.requires_grad = False
    
# List of parameters for both Q-networks (save this for convenience)
q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

# Experience buffer
replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

# Count variables (protip: try to get a feel for how different size networks behave!)
var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

# Set up optimizers for policy and q-function
pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
q_optimizer = Adam(q_params, lr=lr)

def update(data, ac, ac_targ):
    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad()
    loss_q, q_info = compute_loss_q(data, ac, ac_targ)
    loss_q.backward()
    q_optimizer.step()

    # Freeze Q-networks so you don't waste computational effort 
    # computing gradients for them during the policy learning step.
    for p in q_params:
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi, pi_info = compute_loss_pi(data, ac)
    loss_pi.backward()
    pi_optimizer.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_params:
        p.requires_grad = True


    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

def get_action(o, deterministic=False):
    return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                    deterministic)

def test_agent():
    for j in range(num_test_episodes):
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while not(d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time 
            o, r, d, _ = test_env.step(get_action(o, True))
            ep_ret += r
            ep_len += 1
        print("test episode j", j, "return :", ep_ret)

# Prepare for interaction with environment
total_steps = steps_per_epoch * epochs
start_time = time.time()
o, ep_ret, ep_len = env.reset(), 0, 0

# Main loop: collect experience in env and update/log each epoch
for t in range(total_steps):

    # Synchronize worker weights
    ac_state = ac.state_dict()
    ac_targ_state = ac_targ.state_dict()
    set_weights_list = [w.set_weights.remote(ac_state, ac_targ_state) for w in workers]

    # Make rollout and accumulate to Buffers
    ops = [w.rollout.remote(eps_greedy=eps) for w in workers]
    rollout_vals = ray.get(ops)

    for rollout_val in rollout_vals:
        o_buf, a_buf, r_buf, o2_buf, d_buf = rollout_val
        for i in range(max_ep_len):
            o, a, r, o2, d = o_buf[i, :], a_buf[i, :], r_buf[i], o2_buf[i, :], d_buf[i]
            replay_buffer.store(o, a, r, o2, d)
    
    # Update handling
    if t >= update_after and t % update_every == 0:
        print("Updating at", t)
        for j in range(update_every):
            print("Updating...")
            batch = replay_buffer.sample_batch(batch_size)
            update(data=batch, ac=ac, ac_targ=ac_targ)
    
    # End of epoch handling
    if (t+1) % steps_per_epoch == 0:
        epoch = (t+1) // steps_per_epoch

        # Test the performance of the deterministic version of the agent.
        print("Test code to be implemented")
        # test_agent()


    
    # # Until start_steps have elapsed, randomly sample actions
    # # from a uniform distribution for better exploration. Afterwards, 
    # # use the learned policy. 
    # if t > start_steps:
    #     a = get_action(o)
    # else:
    #     a = env.action_space.sample()

    # # Step the env
    # o2, r, d, _ = env.step(a)
    # ep_ret += r
    # ep_len += 1

    # # Ignore the "done" signal if it comes from hitting the time
    # # horizon (that is, when it's an artificial terminal signal
    # # that isn't based on the agent's state)
    # d = False if ep_len==max_ep_len else d

    # # Store experience to replay buffer
    # replay_buffer.store(o, a, r, o2, d)

    # # Super critical, easy to overlook step: make sure to update 
    # # most recent observation!
    # o = o2

    # # End of trajectory handling
    # if d or (ep_len == max_ep_len):
    #     print("ep_ret :", ep_ret)
    #     o, ep_ret, ep_len = env.reset(), 0, 0

    # # Update handling
    # if t >= update_after and t % update_every == 0:
    #     for j in range(update_every):
    #         batch = replay_buffer.sample_batch(batch_size)
    #         update(data=batch)

    # # End of epoch handling
    # if (t+1) % steps_per_epoch == 0:
    #     epoch = (t+1) // steps_per_epoch

    #     # Test the performance of the deterministic version of the agent.
    #     test_agent()
