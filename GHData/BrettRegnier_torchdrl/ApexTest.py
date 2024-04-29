from torchdrl.agents.apex.ApexActor import ApexActor
from torchdrl.agents.apex.ApexLearner import ApexLearner
from torchdrl.agents.ApexAgent import ApexAgent
from torchdrl.agents.QLearningAgent import QLearningAgent

import torchdrl.factories.NeuralNetworkFactory as NeuralNetworkFactory
import torchdrl.factories.OptimizerFactory as OptimizerFactory
import torchdrl.factories.SchedulerFactory as SchedulerFactory
import torchdrl.factories.MemoryFactory as MemoryFactory

from torchdrl.learning.dqn.DQL import DQL
from torchdrl.learning.dqn.DoubleDQL import DoubleDQL

from torchdrl.actions.EpsilonGreedy import EpsilonGreedy
import gym

config = {
    "envs": {
        "name": "BoatLeftRight_v0",
        "num": 1,
        "kwargs": {
            "num_participants": 8,
            "num_locked": 0,
            "lr_goal": 0,
            "lr_relax": 0,
            "fb_goal": 30,
            "fb_relax": 0,
            "rower_state_mode": "2D",
            "rowers_state_normalize": False,
            "include_boat_state_weight": False,
            "reward_mode": "sparse"
        }
    },
    "supervised_agent": {
        "device": "cuda",
        "epochs": 120,
        "shuffle": True,
        "batch_size": 512,
        "kwargs": {
            "scheduler_step_type": "epoch",
            "step_window": 500,
            "evaluate_per_print": False,
            "evaluate_amt": 1000,
            "printing_type": "step",
            "print_mode": "verbose"
        },
        "dataset": {
            "train_set_dir": "dataset/db8_leftright/lg",
            "multi_keys": False
        },
        "optimizer": {
            "name": "adam",
            "kwargs": {
                # 0.0005 is maximum val
                "lr": 0.0004,  # 0.0004 best so far
                "weight_decay": 0.0,
                "amsgrad": True
            }
        },
        "scheduler": {
            "name": "StepLR",
            "kwargs": {
                "step_size": 1,
                "gamma": 1
            }
        },
        "model": {
            "sequential": {
                "conv2d": {
                    "filters": [
                        64
                    ],
                    "kernels": [
                        [2, 1]
                    ],
                    "strides": [
                        [1, 1]
                    ],
                    "paddings": [
                        0
                    ],
                    "activations": [
                        "sigmoid"
                    ],
                    "pools": [
                    ],
                    "flatten": True
                },
                "fullyconnected": {
                    "hidden_layers": [
                        1024,
                        512
                    ],
                    "activations": [
                        "sigmoid",
                        "sigmoid",
                    ],
                    "dropouts": [],
                    "out_features": 256,
                    "final_activation": "sigmoid"
                }
            },
            "head": {
                "dueling": {
                    "hidden_layers": [
                    ],
                    "activations": [
                    ],
                    "dropouts": [],
                    "out_features": None,
                    "final_activation": None
                }
            }
        },
    },
    "q_learning_agent": {
        "device": "cuda",
        "type": "DoubleDQL",
        "name": "Cartpole",
        "batch_size": 32, 
        "kwargs": {
            "clip_grad": 10, 
            "gamma": 0.99, 
            "target_update_frequency": 100, 
            "tau": 1.0,
            "step_window": 1,
            "reward_window": 10,
            "reward_goal": 200,
            "max_steps_per_episode": 200,
            "warm_up": 0,
            "train_checkpoint": True,
            "evaluate_checkpoint": False,
            "evaluate_episodes": 100,
            "evaluate_frequency": 10,
            "checkpoint_root": "models/checkpoints",
            "checkpoint_frequency": 10,
            "checkpoint_max_count": 5,
            "visualize": False,
            "visualize_frequency": -1,
            "seed": -1
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
                "name": "aer",
                "kwargs": {
                    "capacity": 10000,
                    "n_step": 1,
                    "gamma": 0.99,
                }
            },
            "internal_memory": {
                "name": "aer",
                "kwargs": {
                    "capacity": 16,
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
    }
}


# reinforcement learning

q_learning_config = config['q_learning_agent']
device = q_learning_config['device']
actors = []
for i in range(5):
    envs = []
    for _ in range(1):
        envs.append(gym.make("CartPole-v0"))

    model = NeuralNetworkFactory.CreateNetwork(
        q_learning_config['model'], envs[0].observation_space, envs[0].action_space, device)
    q_learning_optimizer = OptimizerFactory.CreateOptimizer(
        q_learning_config['optimizer']['name'], (model.parameters(),), q_learning_config['optimizer']['kwargs'])
    q_learning_scheduler = SchedulerFactory.CreateScheduler(
        q_learning_config['scheduler']['name'], q_learning_optimizer, q_learning_config['scheduler']['kwargs'])
    internal_memory = MemoryFactory.CreateMemory(
        q_learning_config['memories']['internal_memory']['name'], envs[0].observation_space, q_learning_config['memories']['internal_memory']['kwargs'])
    memory = MemoryFactory.CreateMemory(
        q_learning_config['memories']['memory']['name'], envs[0].observation_space, q_learning_config['memories']['memory']['kwargs'])

    memory_n_step = None
    if 'memory_n_step' in q_learning_config['memories']:
        memory_n_step = MemoryFactory.CreateMemory(
            q_learning_config['memories']['memory_n_step']['name'], envs[0].observation_space, q_learning_config['memories']['memory_n_step']['kwargs'])

    action_function = EpsilonGreedy(**q_learning_config['action_function']['kwargs'])
    loss_function = DoubleDQL(device)

    q_learning_args = (
        i,
        internal_memory,
        q_learning_config['name'], 
        envs, 
        model,
        action_function,
        loss_function,
        q_learning_optimizer, 
        q_learning_config['batch_size'], 
        memory
    )
    q_learning_kwargs = q_learning_config['kwargs']
    q_learning_kwargs['memory_n_step'] = memory_n_step
    q_learning_kwargs['scheduler'] = q_learning_scheduler
    q_learning_kwargs['device'] = "cpu"

    actors.append(ApexActor(*q_learning_args, **q_learning_kwargs));


envs = []
for _ in range(1):
    envs.append(gym.make("CartPole-v0"))
q_learning_config = config['q_learning_agent']
device = q_learning_config['device']

# reinforcement learning
model = NeuralNetworkFactory.CreateNetwork(
    q_learning_config['model'], envs[0].observation_space, envs[0].action_space, device)
q_learning_optimizer = OptimizerFactory.CreateOptimizer(
    q_learning_config['optimizer']['name'], (model.parameters(),), q_learning_config['optimizer']['kwargs'])
q_learning_scheduler = SchedulerFactory.CreateScheduler(
    q_learning_config['scheduler']['name'], q_learning_optimizer, q_learning_config['scheduler']['kwargs'])
memory = MemoryFactory.CreateMemory(
    q_learning_config['memories']['memory']['name'], envs[0].observation_space, q_learning_config['memories']['memory']['kwargs'])

memory_n_step = None
if 'memory_n_step' in q_learning_config['memories']:
    memory_n_step = MemoryFactory.CreateMemory(
        q_learning_config['memories']['memory_n_step']['name'], envs[0].observation_space, q_learning_config['memories']['memory_n_step']['kwargs'])

action_function = EpsilonGreedy(**q_learning_config['action_function']['kwargs'])
loss_function = DoubleDQL(device)

q_learning_args = (
    q_learning_config['name'], 
    envs, 
    model,
    action_function,
    loss_function,
    q_learning_optimizer, 
    q_learning_config['batch_size'], 
    memory
)
q_learning_kwargs = q_learning_config['kwargs']
q_learning_kwargs['memory_n_step'] = memory_n_step
q_learning_kwargs['scheduler'] = q_learning_scheduler
q_learning_kwargs['device'] = device

learner = ApexLearner(*q_learning_args, **q_learning_kwargs)

agent = ApexAgent(learner, actors, 2)
agent.TrainNoYield()