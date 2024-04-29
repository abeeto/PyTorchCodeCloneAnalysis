from tensorflow.contrib.rnn import LSTMCell
from torch import nn
import tensorflow as tf
import torch
import numpy as np

batch_size = 3
t = 2
num_in = 7
num_hidden = 5

lstm_torch = nn.LSTM(
    input_size=num_in,
    hidden_size=num_hidden,
    batch_first=True
)

input_ = np.random.rand(batch_size, t, num_in).astype(np.float32)

input_placeholder = tf.placeholder(tf.float32, input_.shape)
lstm_tf = LSTMCell(num_hidden, forget_bias=0.0)
out_tf_sy, state_tf_sy = tf.nn.dynamic_rnn(lstm_tf, input_placeholder, dtype=tf.float32)

out_torch, state_torch = lstm_torch(torch.tensor(input_))

params = list(lstm_torch.parameters())

# tf kernel dimensions [num_in + num_hidden, num_hidden * 4]
# torch kernel dimensions [num_hidden * 4, num_in + num_hidden]
# so transpose torch kernel
kernel_torch = torch.cat(params[:2], dim=1).detach().numpy().T
bias_torch = (params[2] + params[3]).detach().numpy()

# tf order is i, z, f, o
# torch order is i, f, z, o
reordering = [0, 2, 1, 3]
kernel_torch = np.concatenate(np.array(np.split(kernel_torch, 4, axis=1))[reordering], axis=1)
bias_torch = np.concatenate(np.array(np.split(bias_torch, 4))[reordering])

kernel_tf, bias_tf = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

assign_ops = [
    tf.assign(kernel_tf, tf.constant(kernel_torch)),
    tf.assign(bias_tf, tf.constant(bias_torch)),
]

with tf.Session() as sess:

    sess.run(assign_ops)

    out_tf, state_tf, kernel_val = sess.run(
        [out_tf_sy, state_tf_sy, lstm_tf._kernel],
        {
            input_placeholder: input_
        }
    )

print(np.isclose(out_torch.detach().numpy(), out_tf).all())
