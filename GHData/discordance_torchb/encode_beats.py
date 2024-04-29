from pymongo import MongoClient
from random import shuffle

# internal package
import utils
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

# database
client = MongoClient('localhost', 27017)
db = client.bheat
collection = db.origset

beats = list(collection.find({ '$or':[{'class':6},{'class':3}],
                              'bar': 128,
                              'diversity': {'$gt': 0.08},
                              'gridicity': {'$lt': 0.75},
                              }).limit(100))

# select random
shuffle(beats)

# decompress in numpy
alll = []
for i, beat in enumerate(beats):
    np_beat = utils.decompress(beat['zip'], beat['bar'])
    for j, np_bar in enumerate(np_beat):
        alll.append(np_bar)
alll = np.array(alll)
print "before:", alll.shape

# downsample
alll = utils.map2twelve(alll)
print "down-sampled:", alll.shape

alll = utils.clean_and_unique_beats(alll)
print "after:", alll.shape

alll_f = alll.reshape((alll.shape[0],alll.shape[1]*12,))
# okay
print "dataset: ", alll_f.shape

# manage data loading
train_size = (alll_f.shape[0]/4)*3
val_size = (alll_f.shape[0]/4)

print "train/valid sets size: ", train_size, val_size, "\n"

# Training Parameters
learning_rate = 1e-5
num_steps = 500000
batch_size = 128

display_step = 1000

# Network Parameters
num_hidden_1 = 1024 # 1st layer num features
num_hidden_2 = 256  # 2nd layer num features
num_input = alll_f.shape[1]

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input],  name='input_layer')
# Xcoded = tf.placeholder("float", [None, num_hidden_2],  name='coded_input_layer')

is_training = tf.placeholder(tf.bool, name='is_training')

weights = {
    'encoder_h1': tf.get_variable("encoder_h1", shape=[num_input, num_hidden_1], initializer=tf.contrib.layers.xavier_initializer()),
    'encoder_h2': tf.get_variable("encoder_h2", shape=[num_hidden_1, num_hidden_2], initializer=tf.contrib.layers.xavier_initializer()),
    'decoder_h1': tf.get_variable("decoder_h1", shape=[num_hidden_2, num_hidden_1], initializer=tf.contrib.layers.xavier_initializer()),
    'decoder_h2': tf.get_variable("decoder_h2", shape=[num_hidden_1, num_input], initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'encoder_b1': tf.get_variable("encoder_b1", shape=[num_hidden_1], initializer=tf.zeros_initializer()),
    'encoder_b2': tf.get_variable("encoder_b2", shape=[num_hidden_2], initializer=tf.zeros_initializer()),
    'decoder_b1': tf.get_variable("decoder_b1", shape=[num_hidden_1], initializer=tf.zeros_initializer()),
    'decoder_b2': tf.get_variable("decoder_b2", shape=[num_input], initializer=tf.zeros_initializer()),
}

# Building the encoder
def encoder(x):
    # Encoder Input layer
    l1_op = tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1'])
    layer_1 = selu(l1_op)
    layer_1 = tf.layers.dropout(layer_1, 0.33, training=is_training)

    # Encoder Hidden layer
    l2_op = tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2'])
    layer_2 = tf.nn.sigmoid(l2_op)

    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer
    l1_op = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])
    layer_1 = selu(l1_op)
    layer_1 = tf.layers.dropout(layer_1, 0.33, training=is_training)

    # Decoder Out layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2'])
    # clamp
    # layer_2 = tf.clip_by_value(layer_2, 0.0, 1.0)
    return layer_2


# Construct model
encoder_op = tf.identity(encoder(X), name="encoder_op")
decoder_op = tf.identity(decoder(encoder_op), name="decoder_op")

normalized_output = tf.identity(tf.clip_by_value(decoder_op, 0, 1), name="normalized_output")

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

#check accuracy
correct_prediction = tf.equal(tf.argmax(y_true,1), tf.argmax(y_pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define loss and optimizer
def loss_func():
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(cross_ent)*10 #scale

loss = loss_func()
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]

    return np.asarray(data_shuffle)

# saver
saver = tf.train.Saver()


# create log file
fl = open("logs/autoencoder.csv", "a+")
fl.write("it, train, val, acc \n")
fl.close()
# Start Training
# Start a new TF session
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x = next_batch(batch_size, alll_f[:train_size])
        val_x = next_batch(batch_size, alll_f[train_size:])

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, is_training: 1})
        vl = sess.run(loss, feed_dict={X: val_x, is_training: 0})
        acc = sess.run(accuracy, feed_dict={X: val_x, is_training: 0})

        if i % display_step == 0:
            saver.save(sess, "encoded/jazz1/model_%i_%f.tb"% (i, l))

        if i % (display_step/32) == 0:
            print('Step %i: Minibatch Loss: %f, Acc: %f, Valid Loss: %f' % (i, l, acc, vl))
            #
            fl = open("logs/autoencoder.csv", "a+")
            fl.write("%i, %f, %f, %f \n"%(i, l, vl, acc))
            fl.close()

        if i % (display_step) == 0:
            e = sess.run(encoder_op, feed_dict={X: val_x, is_training: 0})
            g = sess.run(decoder_op, feed_dict={X: val_x, is_training: 0})
            print "ORIG: \n"
            print utils.draw(val_x[0].reshape((128,12)))
            print val_x[0].reshape((128,12))
            print "REBUILD: \n"
            print utils.draw(g[0].reshape((128,12))) + "\n"
            print g[0].reshape((128,12))
            print g[0].reshape((128,12))[0][8:]
            # print e[0]
