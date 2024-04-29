# This example illustrates the difference between tensorflow and torch in terms of static and dynamic graph structures. 
# TF uses a static computational graph structure, whereas Torch uses a dynamic computational graph structure. There are advantages and disadvantages to both. For details see this nice post: https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b
# Here we use tensorflow to implement the same autograd function we implemented in example4 using torch

import tensorflow as tf
import numpy as np


# First we set up the computational graph:

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create placeholders for the input and target data; these will be filled with real data when we execute the graph.
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# Create variables for the weights and initialize them with random data.
# A TensorFlow variable persists its value across executions of the graph.
w1 = tf.Variable(tf.random_normal((D_in,H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# Forward pass: Unlike Torch, in TF, these following operations do not perform any mathematical operation rightaway. They merely set up the sequence of operations and these are run when a TF session is called later on.

h = tf.matmul(x,w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)


# Compute the loss using operations on TensorFlow Tensors

loss = tf.reduce_sum((y - y_pred) ** 2.0)

# Compute gradient of the loss with respect to w1 and w2

grad_w1, grad_w2 = tf.gradients(loss, [w1,w2])

# Update the weights using gradient descent. To actually update the weights
# we need to evaluate new_w1 and new_w2 when executing the graph. Note that
# in TensorFlow the the act of updating the value of the weights is part of
# the computational graph; in PyTorch this happens outside the computational
# graph.

learning_rate =1e-6

new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# Now we have built our computational graph, so we enter a TensorFlow session to
# actually execute the graph.

with tf.Session() as sess:
	# Run the graph once to initialize the variables w1 and w2
	sess.run(tf.global_variables_initializer())
	
	# Create nump arrays holding the actual data for the inputs x and targets y
	x_value = np.random.randn(N, D_in)
	y_value = np.random.randn(N, D_out)
	for _ in range(500):
		# Execute the graph many times. Each time it executes we want to bind
        	# x_value to x and y_value to y, specified with the feed_dict argument.
        	# Each time we execute the graph we want to compute the values for loss,
        	# new_w1, and new_w2; the values of these Tensors are returned as numpy
        	# arrays.
		loss_value, _, _ = sess.run([loss, new_w1, new_w2], feed_dict={x: x_value, y: y_value})
		print(loss_value)


