"""
Tensorflow implementation for MNIST
"""
import time
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from weight_gen import get_tf_weights


def model_builder(x, weights):
    x = tf.reshape(x, [-1, 28, 28, 1])

    conv1_w, conv1_b = weights[0], weights[1]
    Wv = tf.get_variable('conv1_w', shape=conv1_w.shape,
                         initializer=tf.constant_initializer(conv1_w))
    bv = tf.get_variable('conv1_b', shape=conv1_b.shape,
                         initializer=tf.constant_initializer(conv1_b))
    x = tf.nn.conv2d(x, Wv, (1, 1, 1, 1), 'VALID')
    x = tf.nn.bias_add(x, bv)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    x = tf.nn.relu(x)

    conv2_w, conv2_b = weights[2], weights[3]
    Wv = tf.get_variable('conv2_w', shape=conv2_w.shape,
                         initializer=tf.constant_initializer(conv2_w))
    bv = tf.get_variable('conv2_b', shape=conv2_b.shape,
                         initializer=tf.constant_initializer(conv2_b))
    x = tf.nn.conv2d(x, Wv, (1, 1, 1, 1), 'VALID')
    x = tf.nn.bias_add(x, bv)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    x = tf.nn.relu(x)
    x = tf.reshape(x, [-1, 320])

    fc1_w, fc1_b = weights[4], weights[5]
    Wv = tf.get_variable('fc1_w', shape=fc1_w.shape,
                         initializer=tf.constant_initializer(fc1_w))
    bv = tf.get_variable('fc1_b', shape=fc1_b.shape,
                         initializer=tf.constant_initializer(fc1_b))
    x = tf.matmul(x, Wv) + bv
    x = tf.nn.dropout(tf.nn.relu(x), 0.5)

    fc2_w, fc2_b = weights[6], weights[7]
    Wv = tf.get_variable('fc2_w', shape=fc2_w.shape,
                         initializer=tf.constant_initializer(fc2_w))
    bv = tf.get_variable('fc2_b', shape=fc2_b.shape,
                         initializer=tf.constant_initializer(fc2_b))

    x = tf.matmul(x, Wv) + bv
    return x


data = input_data.read_data_sets("./tf_data/", one_hot=True)


ipt_x = tf.placeholder(tf.float32, [None, 28 * 28])
ipt_y = tf.placeholder(tf.float32, [None, 10])
weights = get_tf_weights()


conv_y = model_builder(ipt_x, weights)


loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=ipt_y, logits=conv_y)
)
train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


start = time.time()


NUM_ITER = 10000


for i in range(NUM_ITER):
    batch_x, batch_y = data.train.next_batch(64)
    if i % 10 == 0:
        loss_val = loss.eval(feed_dict={ipt_x: batch_x, ipt_y: batch_y},
                             session=sess)
        print('Loss at iteration %04d is %.05f' % (i, loss_val))
    train_step.run(feed_dict={ipt_x: batch_x, ipt_y: batch_y}, session=sess)


print('Time for %d iterations is: %s' % (NUM_ITER, time.time() - start))

sess.close()
