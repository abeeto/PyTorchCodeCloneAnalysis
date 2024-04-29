from tensorflow.contrib.rnn import LSTMCell
from torch import nn
import tensorflow as tf
from scipy.stats import logistic
import numpy as np

batch_size = 1
t = 2
num_in = 1
num_hidden = 1

lstm_torch = nn.LSTM(
    input_size=num_in,
    hidden_size=num_hidden,
    batch_first=True
)

input_ = np.ones((batch_size, t, num_in)).astype(np.float32)

input_placeholder = tf.placeholder(tf.float32, input_.shape)
lstm_tf = LSTMCell(num_hidden, forget_bias=0.0)
out_tf_sy, state_tf_sy = tf.nn.dynamic_rnn(lstm_tf, input_placeholder, dtype=tf.float32)

kernel_tf, bias_tf = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

print(kernel_tf.shape, bias_tf.shape)

kernel_np = np.arange(1, 9).astype(np.float32).reshape(2, 4) / 8
bias_np = np.arange(1, 5).astype(np.float32) / 4

assign_ops = [
    tf.assign(kernel_tf, tf.constant(kernel_np)),
    tf.assign(bias_tf, tf.constant(bias_np)),
]

with tf.Session() as sess:

    sess.run(assign_ops)

    out_tf, state_tf = sess.run(
        [out_tf_sy, state_tf_sy],
        {
            input_placeholder: input_
        }
    )

    print(out_tf[:, :, 0])


def calc_gate(x_curr, y_prev, w, r, b, func=logistic.cdf):

    return func(w * x_curr + r * y_prev + b)


x = 1
y_ = 0
c_ = 0

i = calc_gate(x, y_, kernel_np[0, 0], kernel_np[1, 0], bias_np[0])
z = calc_gate(x, y_, kernel_np[0, 1], kernel_np[1, 1], bias_np[1], func=np.tanh)
f = calc_gate(x, y_, kernel_np[0, 2], kernel_np[1, 2], bias_np[2])
o = calc_gate(x, y_, kernel_np[0, 3], kernel_np[1, 3], bias_np[3])

c = i * z + f * c_
y = o * np.tanh(c)

print(y)

c_ = c
y_ = y

i = calc_gate(x, y_, kernel_np[0, 0], kernel_np[1, 0], bias_np[0])
z = calc_gate(x, y_, kernel_np[0, 1], kernel_np[1, 1], bias_np[1], func=np.tanh)
f = calc_gate(x, y_, kernel_np[0, 2], kernel_np[1, 2], bias_np[2])
o = calc_gate(x, y_, kernel_np[0, 3], kernel_np[1, 3], bias_np[3])

c = i * z + f * c_
y = o * np.tanh(c)

print(y)
