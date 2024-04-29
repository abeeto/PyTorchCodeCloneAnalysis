import random

import torch
import numpy as np 
import gym

from torchdrl.managers.RLManager import RLManager
import torchdrl.factories.AgentFactory as AgentFactory

config = {
    "env": {
        "name": "CartPole-v0",
        "num": 1,
        "kwargs": {
        }
    },
    "q_learning_agent": {
        "type": "DoubleDQL",
        "name": "CartPole_v4",
        "batch_size": 32, 
        "kwargs": {
            "clip_grad": 10, 
            "gamma": 0.99, 
            "target_update_frequency": 200, 
            "tau": 1.0,
            "max_steps_per_episode": 200,
            "seed": 0, # REMOVE move it as a top level config
            "warm_up": -1,
            "device": "cuda",
        },
        "action_function": {
            "name": "EpsilonGreedy", 
            "kwargs": {
                "epsilon": 1.0,
                "epsilon_decay": 0.99,
                "epsilon_min": 0.01,
            }
        },
        "optimizer": {
            "name": "adam",
            "kwargs": {
                "lr": 0.0001, 
            }
        },
        "scheduler": {
            "name": "StepLR",
            "kwargs": {
                "step_size": 5,
                "gamma": 0.6
            }
        },
        "memories": {
            "memory": {
                "name": "ER",
                "kwargs": {
                    "capacity": 10000,
                    "n_step": 1,
                    "gamma": 0.99,
                }
            }
        },
        "model": {
            "sequential": {
                "fullyconnected": {
                    "hidden_layers": [
                        1024
                    ],
                    "activations": [
                        "relu"
                    ],
                    "dropouts": [],
                    "out_features": 1024,
                    "final_activation": "relu"
                }
            },
            "head": {
                "dueling": {
                    "hidden_layers": [
                        1024,
                    ],
                    "activations": [
                        "relu"
                    ],
                    "dropouts": [],
                    "out_features": None,
                    "final_activation": None, # using softmax has really stable learning, but doesn't increase much
                }
            }
        }
    },
    "manager": {
        "name": "rlmanager",
        "kwargs": {
            "metrics": {

            },
            "step_window": 1,
            "reward_window": 10,
            "reward_goal": 195,
            "train_checkpoint": True,
            "evaluate_checkpoint": True,
            "evaluate_episodes": 1,
            "evaluate_frequency": 5,
            "checkpoint_root": "models/checkpoints",
            "checkpoint_frequency": 10,
            "checkpoint_max_count": 5,
            "record_chart_metrics": True,
            "visualize": False,
            "visualize_frequency": -1,
        }
    }
}

# TODO move into a factory
envs = []
for i in range(1):
    envs.append(gym.make("CartPole-v0"))

    

# set the seed
seed = 0
if seed >= 0:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Seed envs
    for i, env in enumerate(envs):
        env.seed(seed+i)

q_learning_agent = AgentFactory.CreateQLearningAgent(config['q_learning_agent'], envs)

manager = RLManager(q_learning_agent, **config['manager']['kwargs'])

manager.TrainNoYield()
