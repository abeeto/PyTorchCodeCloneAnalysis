from collections import OrderedDict

import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from autoencoder import Autoencoder

class LTAExtractor(BaseFeaturesExtractor):
    def __init__(
        self, 
        observation_space: gym.spaces.Box, 
        features_dim: int, 
        n_actions: int,
        human_pred: bool = True,
        strategy_encoder: Autoencoder = None
    ):
        super(LTAExtractor, self).__init__(observation_space, features_dim)
        self.human_pred = human_pred

        input_shape = sum(observation_space.nvec) # input vector shape is one-hot encoded
        human_out = n_actions
        robot_out = 32

        self.strategy_encoder = strategy_encoder

        if self.strategy_encoder is not None:
            input_shape = input_shape + strategy_encoder.n_latent
            # freeze all layers of encoder
            for p in strategy_encoder.parameters():
                p.requires_grad = False

        # TODO: include strategy encoder in custom forward of human model? may make more
        # sense than having to manually process observations everywhere
        self.human = nn.Sequential(OrderedDict([
            ("human_linear0", nn.Linear(input_shape, 32)),
            ("human_relu0", nn.ReLU()),
            ("human_linear1", nn.Linear(32, human_out)),
            # ("human_softmax1", nn.Softmax(dim=1)) # softmax for action probabilities
        ]))
 
        self.robot = nn.Sequential(OrderedDict([
            ("robot_linear0", nn.Linear(input_shape, 32)),
            ("robot_relu0", nn.ReLU()),
            ("robot_linear1", nn.Linear(32, robot_out)),
            ("robot_relu1", nn.ReLU())
        ]))

        if self.human_pred:
            joint_in = human_out + robot_out
        else:
            joint_in = robot_out

        self.joint = nn.Sequential(OrderedDict([
            ("joint_linear0", nn.Linear(joint_in, features_dim)),
            ("joint_relu0", nn.ReLU())
        ]))


    def forward(self, observations: th.Tensor) -> th.Tensor:
        # concat human and robot outputs into joint reasoning
        if self.human_pred:

            # append latent state of autoencoder to observation
            if self.strategy_encoder is not None:
                latent_state = self.strategy_encoder.encoder(observations[:,4:])
                observations = th.cat((observations, latent_state), 1)

            human_pred = self.human(observations)
            robot_feat = self.robot(observations)
            # return self.joint(th.cat((human_pred, robot_feat), 1))
            return th.cat((human_pred, robot_feat), 1)
        else:
            # return self.joint(self.robot(observations))
            return self.robot(observations)
