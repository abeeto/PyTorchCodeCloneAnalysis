"""Small test script that analysis if there is a speed difference between pytorch and
tensorflow when using rsample/sample on the normal distribution.
"""

import timeit

# Script settings
N_SAMPLE = int(1e5)  # How many times we sample

######################################################
# Time sample action #################################
######################################################
print("====Sample speed comparison Pytorch/Tensorflow====")
print(
    f"PYTORCH: Analysing the speed of sampling {N_SAMPLE} times from the normal "
    "distribution..."
)

# Time pytroch sample action
pytorch_setup_code = """
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
# torch.set_default_tensor_type('torch.cuda.FloatTensor') # Enable global GPU
# torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
# torch.backends.cudnn.fastest = True  # Enable cudnn fastest autotuner
batch_size=256
mu = torch.zeros(batch_size, 3)
std = torch.ones(batch_size, 3)
"""
pytorch_sample_code = """
normal_distribution = Normal(torch.zeros(batch_size, 3), torch.ones(batch_size, 3))
pi_action = (
    normal_distribution.sample()
)  # Sample while using the parameterization trick
"""
print("Pytorch test...")
pytorch_time = timeit.timeit(
    pytorch_sample_code, setup=pytorch_setup_code, number=N_SAMPLE
)

# Tensorflowsample action
tf_setup_code = """
import tensorflow as tf
import tensorflow_probability as tfp
from squash_bijector import SquashBijector
tf.config.set_visible_devices([], "GPU") # Disable GPU
batch_size=256
mu = tf.zeros(3)
std = tf.ones(3)
@tf.function
def sample_function():
    normal_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(3), scale_diag=tf.ones(3)
    )
    epsilon = normal_distribution.sample(batch_size)
"""
tf_sample_code = """
sample_function()
"""
print("Tensorflow test...")
tf_time = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)

######################################################
# Time rsample action ################################
######################################################
print(
    f"Analysing the speed of r-sampling {N_SAMPLE} times from the normal "
    "distribution..."
)

# Time pytroch sample action
pytorch_setup_code = """
import torch
from torch.distributions.normal import Normal
torch.set_default_tensor_type('torch.cuda.FloatTensor') # Enable global GPU
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
# torch.backends.cudnn.fastest = True  # Enable cudnn fastest autotuner
batch_size=256
mu = torch.zeros(batch_size, 3)
std = torch.ones(batch_size, 3)
"""
pytorch_sample_code = """
normal_distribution = Normal(torch.zeros(batch_size, 3), torch.ones(batch_size, 3))
pi_action = (
    normal_distribution.rsample()
)  # Sample while using the parameterization trick
"""
print("Pytorch test...")
pytorch_time_2 = timeit.timeit(
    pytorch_sample_code, setup=pytorch_setup_code, number=N_SAMPLE
)

# Tensorflowsample action
tf_setup_code = """
import tensorflow as tf
import tensorflow_probability as tfp
# tf.config.set_visible_devices([], "GPU") # Disable GPU
batch_size=256
mu = tf.zeros((batch_size, 3), dtype=tf.float32)
std = tf.ones((batch_size, 3), dtype=tf.float32)
@tf.function
def sample_function():
    affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(std))
    normal_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(3), scale_diag=tf.ones(3)
    )
    epsilon = normal_distribution.sample(batch_size)
    raw_action = affine_bijector.forward(epsilon)
"""
tf_sample_code = """
sample_function()
"""
print("Tensorflow test...")
tf_time_2 = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)

######################################################
# Print results ######################################
######################################################
# print("\nTest tensorflow/pytorch sample method speed:")
print(f"- Pytorch_sample_time: {pytorch_time} s")
print(f"- Tf_sample_time: {tf_time} s")
print("\nTest tensorflow/pytorch sample method speed:")
print(f"- Pytorch_rsample_time: {pytorch_time_2} s")
print(f"- Tf_rsample_time: {tf_time_2} s")
