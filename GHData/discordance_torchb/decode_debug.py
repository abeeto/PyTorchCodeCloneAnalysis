# internal package
import utils
import numpy as np
import tensorflow as tf


# load the generated

generated = np.load("generated/jazz1/gen990000.000000.npy")
witness = np.load("encoded/jazz1.npy")
np.random.shuffle(witness)


# will load the model
print "load the model"
with tf.Session() as sess:
    with tf.device("/cpu:0"):

        saver = tf.train.import_meta_graph('encoded/jazz1/model_25000_0.075606.tb.meta')
        saver.restore(sess, tf.train.latest_checkpoint('encoded/jazz1/'))
        graph = tf.get_default_graph()

        is_training = graph.get_tensor_by_name("is_training:0")

        CodedX = tf.placeholder("float", [None, 256],  name='coded_input_layer')

        # remake the decoder
        dec_h1 = graph.get_tensor_by_name("decoder_h1:0")
        dec_h2 = graph.get_tensor_by_name("decoder_h2:0")
        dec_b1 = graph.get_tensor_by_name("decoder_b1:0")
        dec_b2 = graph.get_tensor_by_name("decoder_b2:0")

        # Decoder Hidden layer
        l1_op = tf.add(tf.matmul(CodedX, dec_h1), dec_b1)
        layer_1 = tf.nn.relu(l1_op)

        # Decoder Out layer
        layer_2 = tf.add(tf.matmul(layer_1, dec_h2), dec_b2)
        layer_2 = tf.clip_by_value(layer_2, 0, 1)
        result = sess.run(layer_2, feed_dict={CodedX:generated,is_training:0})
        for e in result:
            print utils.draw(e.reshape((128,20))), e
        print result.shape
