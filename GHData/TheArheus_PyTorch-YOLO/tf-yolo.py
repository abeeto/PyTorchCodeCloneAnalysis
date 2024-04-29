import tensorflow as tf

gf = tf.compat.v1.GraphDef()
data = open("saved_model.pb", "rb")

tf_d = gf.ParseFromString(data.read())
print(tf_d)