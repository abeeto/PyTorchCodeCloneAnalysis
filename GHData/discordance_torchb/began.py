import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import ops


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

mb_size = 64
X_dim = 128
z_dim = 64
h_dim = 256
lr = 1e-4
m = 5
lam = 1e-3
gamma = 0.5
k_curr = 0

# load data
#
encoded_data = np.load("encoded/jazz1.npy")

print "loaded data", encoded_data.shape
normalized = (encoded_data-np.min(encoded_data))/(np.max(encoded_data)-np.min(encoded_data))

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim], name="input_layer")
z = tf.placeholder(tf.float32, shape=[None, z_dim], name="noise_layer")
k = tf.placeholder(tf.float32, name="k_layer")

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]), name="d_w1") # layer 1 weight discriminator
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name="d_b1") # layer 1 bias discriminator
D_W2 = tf.Variable(xavier_init([h_dim, 1]), name="d_w2") # layer 2 weight discriminator
D_b2 = tf.Variable(tf.zeros(shape=[1]), name="d_b2") # layer 2 bias discriminator

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]), name="g_w1") # layer 1 weight generator
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name="g_b1") # layer 1 bias generator
G_W2 = tf.Variable(xavier_init([h_dim, X_dim]), name="g_w2") # layer 2 weight generator
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]), name="g_b2") # layer 2 bias generator

theta_G = [G_W1, G_W2, G_b1, G_b2]
theta_D = [D_W1, D_W2, D_b1, D_b2]

def sample_z(m, n):
    return np.random.uniform(0., 1., size=[m, n])


def generator(z):
    G_h1 = selu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = selu(tf.matmul(x, D_W1) + D_b1)
    D_h1 = tf.layers.dropout(D_h1, 0.25, training=True)
    X_recon = tf.matmul(D_h1, D_W2) + D_b2
    return tf.reduce_mean(tf.reduce_sum((x - X_recon)**2, 1))


G_sample = tf.identity(generator(z), name="generator")

D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = D_real - k*D_fake
G_loss = D_fake

tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)


D_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]

    return np.asarray(data_shuffle)

writer = tf.summary.FileWriter("logs/", graph=tf.get_default_graph())
summary_op = tf.summary.merge_all()

# :D
for it in range(1000000):

    X_mb = next_batch(mb_size, normalized)
    _, D_real_curr, summary = sess.run(
        [D_solver, D_real, summary_op],
        feed_dict={X: X_mb, z: sample_z(mb_size, z_dim), k: k_curr}
    )

    _, D_fake_curr = sess.run(
        [G_solver, D_fake],
        feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
    )

    # write log
    writer.add_summary(summary, it)

    X_mb = next_batch(mb_size, normalized)
    z_mb = sample_z(mb_size, z_dim)
    k_curr = k_curr + lam * (gamma*D_real_curr - D_fake_curr)

    if it % 1000 == 0:
        measure = D_real_curr + np.abs(gamma*D_real_curr - D_fake_curr)

        print('{}; Convergence: {:.4}, D_loss: {:.4}; G_loss: {:.4}'.format(it, measure, D_real_curr, D_fake_curr))
        samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})
        np.save("generated/jazz1/gen%f.npy"%it, samples*np.max(encoded_data))
