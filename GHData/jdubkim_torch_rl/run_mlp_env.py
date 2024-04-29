import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Config
parser = argparse.ArgumentParser(description='DQN with PyTorch')
parser.add_argument('--network', type=str, default='mlp', help='Network Type')
parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment')
parser.add_argument('--algo', type=str, default='dqn', help='RL algorithm to run')
parser.add_argument('--mode', type=str, default='train', help='Train or evaluation')
parser.add_argument('--render', action='store_true', default=False, help='Render environment')
parser.add_argument('--load', type=str, default=None, help='Load model')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--iterations', type=int, default=500, help='Number of iterations for training')
parser.add_argument('--eval_per_train', type=int, default=50, help='Evaluate every n training episodes')
parser.add_argument('--max_step', type=int, default=500, help='Max timestep for episode')
parser.add_argument('--threshold_return', type=int, default=100, help='')
parser.add_argument('--tensorboard', action='store_true', default=True)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--DEBUG', type=bool, default=False, help='Debug flag')
args = parser.parse_args()

device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

if args.algo == 'dqn':
    from agents.dqn import DQNAgent


def main():
    # Initialise Environment
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_num = env.action_space.n

    print('-----------------------------------')
    print('Environment  : ', args.env)
    print('Algorithm    : ', args.algo)
    print('State dim    : ', obs_dim)
    print('Action number: ', act_num)
    print('-----------------------------------')

    # Set random seed
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create a RL agent
    agent = DQNAgent(env, args, device, obs_dim, act_num)

    # Load model if saved model exists
    if args.load is not None:
        pretrained_model_path = os.path.join('./saved_mode/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path, map_location=device)

        if args.algo == 'dqn':
            agent.qf.load_state_dict(pretrained_model)

    # Create a Summary object
    if args.tensorboard and args.load is None:
        dir_name = 'runs/' + args.env + '/' \
                   + args.algo + '_s_' + str(args.seed) \
                   + '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        writer = SummaryWriter(log_dir=dir_name)

    start_time = time.time()

    train_num_steps = 0
    train_sum_returns = 0.
    train_num_episodes = 0
    eval_sum_returns = 0.
    eval_num_episodes = 0

    for i in range(args.iterations):

        if args.mode == 'train':
            agent.eval_mode = False

            train_step_length, train_episode_return = agent.run(args.max_step)

            train_num_steps += train_step_length
            train_sum_returns += train_episode_return
            train_num_episodes += 1

            train_average_return = train_sum_returns / train_num_episodes

            if args.tensorboard and args.load is None:
                writer.add_scalar('Train/AverageReturns', train_average_return, i)
                writer.add_scalar('Train/AverageReturns', train_episode_return, i)

        # Evaluation
        if (i + 1) % args.eval_per_train == 0:
            agent.eval_mode = True

            for _ in range(args.max_step):
                eval_step_length, eval_episode_return = agent.run(args.max_step)

                eval_sum_returns += eval_episode_return
                eval_num_episodes += 1

            eval_average_return = eval_sum_returns / eval_num_episodes

            if args.tensorboard and args.load is None:
                writer.add_scalar('Eval/AverageReturns', eval_average_return, i)
                writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, i)

            if args.mode == 'train':
                print('---------------------------------------')
                print('Steps              : ', train_num_steps)
                print('Episodes           : ', train_num_episodes)
                print('Episode Return     : ', round(train_episode_return, 2))
                print('Average Return     : ', round(train_average_return, 2))
                print('Eval Episodes      : ', eval_num_episodes)
                print('Eval Episode Return: ', round(eval_episode_return, 2))
                print('Eval Average Return: ', round(eval_average_return, 2))
                print('OtherLogs          : ', agent.logger)
                print('Time               : ', int(time.time() - start_time))
                print('---------------------------------------')

                # Save the trained model

                if eval_average_return >= args.threshold_return:
                    if not os.path.exists('./saved_model'):
                        os.mkdir('./saved_model')

                    ckpt_path = os.path.join('./save_model/' + args.env + '_' + args.algo \
                                             + '_s_' + str(args.seed) \
                                             + '_i_' + str(i + 1) \
                                             + '_tr_' + str(round(train_episode_return, 2)) \
                                             + '_er_' + str(round(eval_episode_return, 2)) + '.pt')

                    if args.algo == 'dqn':
                        torch.save(agent.qf.state_dict(), ckpt_path)

            elif args.mode == 'test':
                print('---------------------------------------')
                print('Steps              : ', train_num_steps)
                print('Eval Episodes      : ', eval_num_episodes)
                print('Eval Episode Return: ', round(eval_episode_return, 2))
                print('Eval Average Return: ', round(eval_average_return, 2))
                print('Time               : ', int(time.time() - start_time))
                print('---------------------------------------')


if __name__ == "__main__":
    main()
