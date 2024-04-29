from tester import Tester
from trainer import Trainer
import math
import numpy as np
import matplotlib.pyplot as plt
from agent import DDPGAgent

from config import *
from params import args
#from Env.netEnv import Env
from Env.env import NetEnv

from log import logger

# https://github.com/ShawnshanksGui/DATE_project/tree/master/DRLTE/drlte
# https://github.com/blackredscarf/pytorch-DDPG
# https://zhuanlan.zhihu.com/p/65931777
# https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py


def build_env(args_):
    env2 = NetEnv("/home/newplan/data/inputs/", "Abi_OBL_3_0_test",
                  "Abi", args.MAX_STEP,  3995)
    return env2


env = build_env(args)
state_dim = env.get_state_dim()
action_dim = env.get_action_dim()

logger.info("State DIM: \n{}".format(state_dim))
logger.info("Action DIM:\n{}".format(sum(action_dim)))
# net_env.show_info()

env.reset()

# exit(0)
# env.render()

# install env to the running params

configs = {
    'args': args,
    'env': env,
    'gamma': 0.99,
    'actor_lr': 0.001,
    'critic_lr': 0.01,
    'tau': 0.02,
    'capacity': 10000,
    'batch_size': 32,
    'using_cuda': args.cuda > 0,
}

agent = DDPGAgent(**configs)
# agent.show_model()

if args.RUNNING_TYPE == "train":
    trainer = Trainer(agent, env, configs)
    trainer.train()
elif args.RUNNING_TYPE == "retrain":
    episode, step = agent.load_checkpoint(
        args.CHECKPOINT_DIR, args.CHECKPOINT_START_EPISODE)
    trainer = Trainer(agent, env, configs)
    trainer.train(episode, step)
elif args.RUNNING_TYPE == "test":
    tester = Tester(agent, env, './running_log/model')
    tester.test(True)
else:
    print("unknown running type: ", args.RUNNING_TYPE)
# env.close()
