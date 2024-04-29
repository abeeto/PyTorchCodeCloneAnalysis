import os
import numpy as np
import torch
import roboverse
import tqdm

from algorithms.ebm import EBMTrainer
from parser import build_parser
from utils.data_utils import load_data_from_npy_chaining, plot
from utils.logger import Logger

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def shape_rewards(args, replay_buffer):
    if args.use_positive_rew:
        if set(np.unique(replay_buffer._rewards)).issubset({0, 1}):
            replay_buffer._rewards = replay_buffer._rewards * 6.0
            replay_buffer._rewards = replay_buffer._rewards + 4.0
        assert set(np.unique(replay_buffer._rewards)).issubset(
            set(6.0 * np.array([0, 1]) + 4.0))
    return replay_buffer

def train():
    args = build_parser()
    env = roboverse.make(args.env, transpose_image=True)
    action_dim = env.action_space.low.size
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    np.random.seed(args.seed)
    logger = Logger(args.save_dir, args.env, args.seed, args.steps)
    filename = logger.get_filename()
    buffer = load_data_from_npy_chaining(args, env, 'image')
    buffer = shape_rewards(args, buffer)

    algo = EBMTrainer(args, env, buffer, filename)
    if os.path.isfile(filename+'model.pt') and args.load_model:
        algo.load()

    for step in tqdm.tqdm(range(0, args.ebm_epochs),
                                smoothing=0.1,
                                disable=not args.tqdm):
        
        batch = buffer.random_batch(args.batch_size)
        stats = algo.train(batch)

        if step % args.log_freq == 0:
            logger.log(stats)
            init = np.random.uniform(low=0, high=1, size=batch['observations'].shape)
            samples = algo.sample(init_samples=init, sgld_steps=5*args.sgld_steps)
            plot(algo.filename+str(algo.step)+'.png', samples)

    algo.save()
    env.close()


if __name__=="__main__":
    train()


