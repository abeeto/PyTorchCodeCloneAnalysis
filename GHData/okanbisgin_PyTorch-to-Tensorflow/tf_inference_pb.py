tf_graph = load_pb('./models/mono_dataset.pb')
sess = tf.Session(graph=tf_graph)

output_tensor = tf_graph.get_tensor_by_name('Sigmoid:0')
input_tensor = tf_graph.get_tensor_by_name('input:0')

output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
print(output)