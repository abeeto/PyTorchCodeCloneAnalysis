import argparse
import os
import pathlib
import torch
import numpy as np
import datetime
import pprint

from gym import spaces

from torch.utils.tensorboard import SummaryWriter

from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer

from envs.dmc_env import make_dmc_env
from mpo import MPOPolicy


D4RL_DATASET = pathlib.Path.home() / ".d4rl/datasets"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    env, train_envs, test_envs = make_dmc_env(
        task=args.task,
        seed=args.seed,
        training_num=args.training_num,
        test_num=args.test_num,
        obs_norm=False,
    )

    args.state_shape = env.observation_space.shape or (env.observation_space.n,)
    args.action_shape = env.action_space.shape or (env.action_space.n,)
    args.max_action = env.action_space.high[0]

    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    net_a = Net(
        state_shape=args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=torch.nn.Tanh,
        device=args.device,
    )
    net_c = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        activation=torch.nn.Tanh,
        device=args.device,
    )

    if isinstance(env.action_space, spaces.Discrete):
        from tianshou.utils.net.discrete import Actor, Critic

        actor = Actor(
            preprocess_net=net_a, action_shape=args.action_shape, device=args.device
        ).to(args.device)
        critic = Critic(net_c, device=args.device).to(args.device)
    else:
        from tianshou.utils.net.continuous import ActorProb, Critic

        actor = ActorProb(
            preprocess_net=net_a,
            action_shape=args.action_shape,
            max_action=args.max_action,
            device=args.device,
            unbounded=False,
        ).to(args.device)
        critic = Critic(preprocess_net=net_c, device=args.device).to(args.device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    policy = MPOPolicy(
        args.state_shape[0],
        args.action_shape[0],
        actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        device=args.device,
    )

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "mpo"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    writer = SummaryWriter(log_path)

    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
    )
    pprint.pprint(result)
