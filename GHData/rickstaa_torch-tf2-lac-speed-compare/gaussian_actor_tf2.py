"""Contains a Tensorflow2.x implementation of the Squashed Gaussian Actor.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from squash_bijector import SquashBijector


class SquashedGaussianActor(tf.keras.Model):
    """The squashed gaussian actor network.

    Attributes:
        net (torch.nn.modules.container.Sequential): The input/hidden layers of the
            network.

        mu (torch.nn.modules.linear.Linear): The output layer which returns the mean of
            the actions.

        log_sigma (torch.nn.modules.linear.Linear): The output layer which returns
            the log standard deviation of the actions.
    """

    def __init__(
        self, obs_dim, act_dim, hidden_sizes, log_std_min=-20, log_std_max=2.0,
    ):
        """Constructs all the necessary attributes for the Squashed Gaussian Actor
        object.

        Args:
            obs_dim (int): The dimension of the observation space.

            act_dim (int): The dimension of the action space.

            hidden_sizes (list): Array containing the sizes of the hidden layers.

            log_std_min (int, optional): The minimum log standard deviation. Defaults
                to -20.

            log_std_max (float, optional): The maximum log standard deviation. Defaults
                to 2.0.
        """
        super().__init__()

        # Get class parameters
        self.s_dim = obs_dim
        self.a_dim = act_dim
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        # Create fully connected layers
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=(self.s_dim), name="input"
                )
            ]
        )
        for i, hidden_size_i in enumerate(hidden_sizes):
            self.net.add(
                tf.keras.layers.Dense(
                    hidden_size_i,
                    activation="relu",
                    name="SquashedGaussianActor" + "/l{}".format(i + 1),
                    trainable=True,
                )
            )

        # Create Mu and log sigma output layers
        self.mu = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=hidden_sizes[-1]
                ),
                tf.keras.layers.Dense(
                    act_dim,
                    activation=None,
                    name="SquashedGaussianActor/mu",
                    trainable=True,
                ),
            ]
        )
        self.log_sigma = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=hidden_sizes[-1]
                ),
                tf.keras.layers.Dense(
                    act_dim,
                    activation=None,
                    name="SquashedGaussianActor/log_sigma",
                    trainable=True,
                ),
            ]
        )

    @tf.function
    def call(self, obs):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

        Returns:
            torch.Tensor,  torch.Tensor: The actions given by the policy, the log
            probabilities of each of these actions.
        """

        # Perform forward pass through fully connected layers
        net_out = self.net(obs)

        # Calculate mu and log_sigma
        mu = self.mu(net_out)
        log_sigma = self.log_sigma(net_out)
        log_sigma = tf.clip_by_value(log_sigma, self._log_std_min, self._log_std_max)

        # Perform re-parameterization trick
        sigma = tf.exp(log_sigma)

        # Create bijectors (Used in the re-parameterization trick)
        squash_bijector = SquashBijector()
        affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(sigma))

        # Sample from the normal distribution and calculate the action
        batch_size = tf.shape(input=obs)[0]
        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim)
        )
        epsilon = base_distribution.sample(batch_size)
        raw_action = affine_bijector.forward(epsilon)
        clipped_a = squash_bijector.forward(raw_action)

        # Transform distribution back to the original policy distribution
        reparm_trick_bijector = tfp.bijectors.Chain((squash_bijector, affine_bijector))
        distribution = tfp.distributions.TransformedDistribution(
            distribution=base_distribution, bijector=reparm_trick_bijector
        )
        clipped_mu = squash_bijector.forward(mu)
        return clipped_a, clipped_mu, distribution.log_prob(clipped_a)
