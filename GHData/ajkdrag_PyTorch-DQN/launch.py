import gym
from train import Trainer
from test import Tester
from argparse import ArgumentParser
from utils.config_parser import ConfigParser
from rl_envs.crawler_robot import CrawlingRobotEnv

RENDER = True


def parse_flags():
    parser = ArgumentParser()
    parser.add_argument("--hyps", default=str, help="Path to hyperparams file")
    parser.add_argument("--opts", default=str, help="Path to options file")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main(FLAGS):
    print(FLAGS)
    hyps, opts = ConfigParser.parse_configs(FLAGS.hyps, FLAGS.opts)

    # env = gym.make(opts["env_name"])
    env = CrawlingRobotEnv(horizon=10000, render=True)
    if FLAGS.test:
        obj = Tester(env, opts, hyps)
    else:
        obj = Trainer(env, opts, hyps)

    obj.setup()
    obj.run()
    env.close()


if __name__ == "__main__":
    flags = parse_flags()
    main(flags)
