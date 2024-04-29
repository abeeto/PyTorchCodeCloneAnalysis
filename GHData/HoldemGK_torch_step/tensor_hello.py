import tensorflow as tf
hello=tf.constant('Hello Python and TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
