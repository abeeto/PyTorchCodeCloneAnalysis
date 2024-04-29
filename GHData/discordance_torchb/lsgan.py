import tensorflow as tf
import numpy as np
import os


mb_size = 128
X_dim = 256 # input dim
z_dim = 128 # noise dim
h_dim = 128 # hiden dim
lr = 1e-4
d_steps = 2


# load data
#
encoded_data = np.load("encoded/jazz1.npy")

print "loaded data", encoded_data.shape

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim], name="input_layer")
z = tf.placeholder(tf.float32, shape=[None, z_dim], name="noise_layer")

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]), name="d_w1") # layer 1 weight discriminator
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name="d_b1") # layer 1 bias discriminator
D_W2 = tf.Variable(xavier_init([h_dim, 1]), name="d_w2") # layer 2 weight discriminator
D_b2 = tf.Variable(tf.zeros(shape=[1]), name="d_b2") # layer 2 bias discriminator

G_W1 = tf.Variable(xavier_init([z_dim, h_dim*2]), name="g_w1") # layer 1 weight generator
G_b1 = tf.Variable(tf.zeros(shape=[h_dim*2]), name="g_b1") # layer 1 bias generator
G_W2 = tf.Variable(xavier_init([h_dim*2, X_dim]), name="g_w2") # layer 2 weight generator
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]), name="g_b2") # layer 2 bias generator

theta_G = [G_W1, G_W2, G_b1, G_b2]
theta_D = [D_W1, D_W2, D_b1, D_b2]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h1 = tf.layers.dropout(D_h1, 0.25, training=True)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


G_sample = tf.identity(generator(z), name="generator")

D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = 0.5 * (tf.reduce_mean((D_real - 1)**2) + tf.reduce_mean(D_fake**2))
G_loss = 0.5 * tf.reduce_mean((D_fake - 1)**2)

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
    for _ in range(d_steps):
        X_mb = next_batch(mb_size, encoded_data)
        z_mb = sample_z(mb_size, z_dim)

        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={X: X_mb, z: z_mb}
        )

    X_mb = next_batch(mb_size, encoded_data)
    z_mb = sample_z(mb_size, z_dim)

    _, G_loss_curr, summary = sess.run(
        [G_solver, G_loss, summary_op],
        feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
    )

    # write log
    writer.add_summary(summary, it)

    if it % 10000 == 0:
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))

        samples = sess.run(G_sample, feed_dict={z: sample_z(64, z_dim)})
        # denormalize !!!
        np.save("generated/jazz1/gen%f.npy"%it, samples)
