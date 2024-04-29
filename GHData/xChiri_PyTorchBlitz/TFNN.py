import tensorflow as tf
import numpy as np

# first we set up the computational graph

# N is batch size; D_in is input dimension; H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# create placeholders for the input and target data; these will be filled with real data when we
# execute the graph.
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# create variables for the weights and initialise them with random data.
# a TensorFlow Variable persists its value across executions of the graph.
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# forward pass: compute the predicted y using operation on TensorFlow Tensors. The following piece
# of code does not perform numeric operations; it just sets up the computational graph.
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# compute loss using operations on TensorFlow Tensors
loss = tf.reduce_sum((y - y_pred) ** 2)

# compute gradient of the loss with respect to w1 and w2
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# update the weights using gradient descent. To update the weights we need to evaluate new_w1 and
# new_w2 when executing the graph. In TF, updating the weights is part of the computational graph;
# in PyTorch this happens outside the computational graph.
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# now the computational graph is built, so we enter a TF session to actually execute the graph.
with tf.Session() as sess:
    # run the graph one to initialize Variables w1 and w2
    sess.run(tf.global_variables_initializer())

    # create numpy arrays holding the actual data for the inputs x and targets y
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)
    for _ in range(500):
        # execute the graph many times. Each time it executes we want to bind x_value to x and
        # y_value to y, specified with the feed_dict argument. Each time the graph is executed
        # we want to compute the values for loss, new_w1, new_w2; these are returned as NumPy arrays.
        loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                    feed_dict={x: x_value, y: y_value})
        print(loss_value)