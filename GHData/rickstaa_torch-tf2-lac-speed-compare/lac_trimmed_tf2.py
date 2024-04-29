"""A trimmed down version of my RL algorithm such that I can investigate why the full
training loop is slower in Pytorch compared to Tensorflow 2.x (eager mode).
"""

import tensorflow as tf

from gaussian_actor_tf2 import SquashedGaussianActor
from lyapunov_critic_tf2 import LyapunovCritic

# Script parameters
GAMMA = 0.999
ALPHA = 0.99
ALPHA3 = 0.2
LABDA = 0.99
POLYAK = 5e-3
LR_A = 1e-4
LR_L = 3e-4
LR_LAG = 3e-4
OBS_DIM = 8
ACT_DIM = 3


class LAC(object):
    """The Lyapunov Actor Critic.
    """

    def __init__(self):

        # Create Learning rate placeholders
        self.LR_A = tf.Variable(LR_A, name="LR_A")
        self.LR_L = tf.Variable(LR_L, name="LR_L")
        self.LR_lag = tf.Variable(LR_LAG, name="LR_lag")
        self.polyak = POLYAK
        self.target_entropy = -ACT_DIM  # lower bound of the policy entropy

        # Create Gaussian Actor (GA) and Lyapunov critic (LC) Networks
        self.ga = SquashedGaussianActor(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            hidden_sizes=[64, 64],
            log_std_min=-20,
            log_std_max=2,
        )
        self.lc = LyapunovCritic(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_sizes=[128, 128],
        )

        # Create GA and LC target networks
        # Don't get optimized but get updated according to the EMA of the main
        self.ga_ = SquashedGaussianActor(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            hidden_sizes=[64, 64],
            log_std_min=-20,
            log_std_max=2,
        )
        self.lc_ = LyapunovCritic(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_sizes=[128, 128],
        )
        self.target_init()

        # Create lagrance multiplier placeholders
        self.log_labda = tf.Variable(tf.math.log(LABDA), name="lambda")
        self.log_alpha = tf.Variable(tf.math.log(ALPHA), name="alpha")

        ###########################################
        # Create optimizers #######################
        ###########################################
        self.alpha_train = tf.keras.optimizers.Adam(learning_rate=self.LR_A)
        self.lambda_train = tf.keras.optimizers.Adam(learning_rate=self.LR_lag)
        self.a_train = tf.keras.optimizers.Adam(learning_rate=self.LR_A)
        self.l_train = tf.keras.optimizers.Adam(learning_rate=self.LR_L)

    @tf.function()
    def learn(self, batch):
        """Runs the SGD to update all the optimize parameters.

        Args:
            batch (numpy.ndarray): The batch of experiences.
        """

        # Retrieve state, action and reward from the batch
        bs = batch["s"]  # state
        ba = batch["a"]  # action
        br = batch["r"]  # reward
        bterminal = batch["terminal"]
        bs_ = batch["s_"]  # next state

        # Update target networks
        self.update_target()

        # Get Lyapunov target
        a_, _, _ = self.ga_(bs_)
        l_ = self.lc_([bs_, a_])
        l_target = br + GAMMA * (1 - bterminal) * tf.stop_gradient(l_)

        # Lyapunov candidate constraint function graph
        with tf.GradientTape() as c_tape:

            # Calculate current lyapunov value
            l = self.lc([bs, ba])

            # Calculate L_backup
            l_error = tf.compat.v1.losses.mean_squared_error(
                labels=l_target, predictions=l
            )

        # Actor loss and optimizer graph
        with tf.GradientTape() as a_tape:

            # Calculate current value and target lyapunov multiplier value
            lya_a_, _, _ = self.ga(bs_)
            lya_l_ = self.lc([bs_, lya_a_])

            # Calculate Lyapunov constraint function
            self.l_delta = tf.reduce_mean(lya_l_ - l + ALPHA * br)

            # Calculate log probability of a_input based on current policy
            _, _, log_pis = self.ga(bs)

            # Calculate actor loss
            a_loss = self.labda * self.l_delta + self.alpha * tf.reduce_mean(log_pis)

        # Lagrance multiplier loss functions and optimizers graphs
        with tf.GradientTape() as lambda_tape:
            labda_loss = -tf.reduce_mean(self.log_labda * self.l_delta)

        # Calculate alpha loss
        with tf.GradientTape() as alpha_tape:
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(log_pis + self.target_entropy)
            )  # Trim down

        # Apply lambda gradients
        lambda_grads = lambda_tape.gradient(labda_loss, [self.log_labda])
        self.lambda_train.apply_gradients(zip(lambda_grads, [self.log_labda]))

        # Apply alpha gradients
        alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_train.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        # Apply actor gradients
        a_grads = a_tape.gradient(a_loss, self.ga.trainable_variables)
        self.a_train.apply_gradients(zip(a_grads, self.ga.trainable_variables))

        # Apply critic gradients
        l_grads = c_tape.gradient(l_error, self.lc.trainable_variables)
        self.l_train.apply_gradients(zip(l_grads, self.lc.trainable_variables))

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    @property
    def labda(self):
        return tf.clip_by_value(tf.exp(self.log_labda), 0, 1)

    @tf.function
    def target_init(self):
        # Initializing targets to match main variables
        for pi_main, pi_targ in zip(self.ga.variables, self.ga_.variables):
            pi_targ.assign(pi_main)
        for l_main, l_targ in zip(self.lc.variables, self.lc_.variables):
            l_targ.assign(l_main)

    @tf.function
    def update_target(self):
        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        for pi_main, pi_targ in zip(self.ga.variables, self.ga_.variables):
            pi_targ.assign(self.polyak * pi_targ + (1 - self.polyak) * pi_main)
        for l_main, l_targ in zip(self.lc.variables, self.lc_.variables):
            l_targ.assign(self.polyak * l_targ + (1 - self.polyak) * l_main)
