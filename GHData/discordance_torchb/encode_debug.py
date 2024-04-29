from pymongo import MongoClient
from random import shuffle

# internal package
import utils
import numpy as np
import tensorflow as tf
import time

# database
client = MongoClient('localhost', 27017)
db = client.bheat
collection = db.origset

beats = list(collection.find({ '$or':[{'class':6},{'class':3}],
                              'bar': 128,
                              'diversity': {'$gt': 0.07},
                              }))

# select random
shuffle(beats)
# beats = beats[:20]

# decompress in numpy
alll = []
for i, beat in enumerate(beats):
    np_beat = utils.decompress(beat['zip'], beat['bar'])
    for j, np_bar in enumerate(np_beat):
        alll.append(np_bar)
alll = np.array(alll)
alll = utils.clean_and_unique_beats(alll)
alll_f = alll.reshape((alll.shape[0],alll.shape[1]*20,))
# okay
print "dataset: ", alll_f.shape

# will load the model
print "load the model"
with tf.Session() as sess:
    with tf.device("/cpu:0"):
        saver = tf.train.import_meta_graph('encoded/jazz1/model_25000_0.075606.tb.meta')
        saver.restore(sess, tf.train.latest_checkpoint('encoded/jazz1/'))
        graph = tf.get_default_graph()
        is_training = graph.get_tensor_by_name("is_training:0")
        X = graph.get_tensor_by_name("input_layer:0")
        enc = graph.get_tensor_by_name("encoder_op:0")
        start_time = time.time()
        result = sess.run(enc, feed_dict={X:alll_f,is_training:0})
        elapsed_time = time.time() - start_time
        print result.shape, elapsed_time
        #save
        np.save("encoded/jazz1.npy", result)
