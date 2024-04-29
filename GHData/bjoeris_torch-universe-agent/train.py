import argparse
import ctypes
import time
import os
from collections import OrderedDict
from collections import defaultdict

import cv2  # hack: unused, but this must load before torch
import universe
from torch import multiprocessing
from torch import optim
import torch

from a3cworker import A3CWorker
from env import create_atari_env
from model import Model
from visualizer import Visualizer

universe.configure_logging()

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num-workers', default=1, type=int,
                    help="Number of workers")
parser.add_argument('-m', '--num-envs', default=1, type=int,
                    help="Number of environments to run on each worker")
parser.add_argument('-e', '--env-id', type=str, default="PongDeterministic-v3",
                    help="Environment id")
parser.add_argument('-l', '--log-dir', type=str, default="/tmp/pong",
                    help="Log directory path")
parser.add_argument('--visualise', action='store_true',
                    help="Visualise the gym environment by running env.render() between each timestep")

def load_checkpoint(model, optimizer, global_steps, checkpoint_dir, checkpoint_id=None):
    r"""Load the model weights from a file"""
    if checkpoint_id is None:
        if not os.path.exists(checkpoint_dir):
            return False
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if len(checkpoint_files) == 0:
            return False
        checkpoint_ids = [int(os.path.splitext(f)[0]) for f in checkpoint_files]
        checkpoint_id = max(checkpoint_ids)
    checkpoint_path = os.path.join(checkpoint_dir, '{}.ckpt'.format(checkpoint_id))
    state_dict = torch.load(checkpoint_path)

    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    optimizer.state = defaultdict(dict)  # bug workaround in Optimizer.load_state_dict
    global_steps.value = state_dict['global_steps']

def save_checkpoint(model, optimizer, global_steps, checkpoint_dir):
    r"""Save a checkpoint file, to recover current training state"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise IOError("Checkpoint directory path is not a directory")
    path = os.path.join(checkpoint_dir, '{}.ckpt'.format(global_steps.value))
    state_dict = OrderedDict()
    state_dict['model'] = model.state_dict()
    state_dict['optimizer'] = optimizer.state_dict()
    state_dict['global_steps'] = global_steps.value
    torch.save(state_dict, path)

def main():
    multiprocessing.set_start_method('forkserver')
    args = parser.parse_args()
    env = lambda: create_atari_env(args.env_id)
    env0 = env()
    model = Model(env0.observation_space.shape, env0.action_space.n,
                  is_cuda=True)
    terminate = multiprocessing.Value(ctypes.c_bool, False)
    global_steps = multiprocessing.Value(ctypes.c_int64, 0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    load_checkpoint(model, optimizer, global_steps, checkpoint_dir)
    workers = [A3CWorker(environment=env(),
                         model=model,
                         log_dir=args.log_dir,
                         terminate=terminate,
                         global_steps=global_steps,
                         optimizer=optimizer,
                         worker_id=i)
               for i in range(args.num_workers)]
    visualizer = Visualizer(env=env0,
                            model=model,
                            terminate=terminate)
    model.share_memory()

    # workers = [multiprocessing.Process(target=a3c.run, kwargs={'optimizer': optimizer,
    #                                                            'worker_id': i})
    #            for i in range(args.num_workers)]
    visualizer_process = multiprocessing.Process(target=visualizer.run)
    visualizer_process.start()
    try:
        while True:
            for w in workers:
                w.step()
    except KeyboardInterrupt:
        terminate.value = True
        visualizer_process.join()


if __name__ == '__main__':
    main()