import numpy as np
import tensorflow as tf
from dataset import load_text_dataset
from tensorflow.contrib import rnn
from util import plot_confusion_matrix
import matplotlib.pyplot as plt

tf.random.set_random_seed(1212)

NUM_LSTM_UNITS = 1
NUM_EPOCHS = 25
BATCH_SIZE = 64
LR = 0.01

def lstm_model_fn(input_x, vocab_size, embed_length):
    # LSTM
	embeded_matrix = np.random.random([vocab_size, embed_length])
	embeded_matrix = np.argsort(embeded_matrix, axis=1)

	tf_embedding = tf.constant(embeded_matrix, dtype=tf.float32)
	embeded_data = tf.nn.embedding_lookup(tf_embedding, input_x)
	
	lstm_layer = rnn.BasicLSTMCell(NUM_LSTM_UNITS,forget_bias=1.0)
	lstm_output,lstm_state = tf.nn.dynamic_rnn(cell=lstm_layer, inputs=tf.transpose(embeded_data, perm=[1, 0, 2]),dtype=tf.float32)


	last = tf.gather(lstm_output, int(lstm_output.get_shape()[0]) - 1)
	
	dense = tf.layers.dense(inputs=last, units=50)
	# dense = tf.layers.dense(inputs=last, units=50, activation=tf.nn.relu)
	return dense

def main():

	# Load dataset
	seq_length = 20
	embed_length = 5
	train_data, test_data, vocab_size = load_text_dataset('datasets/questions/', seq_length)
	train_x, train_y = train_data
	test_x, test_y = test_data
	# import pdb; pdb.set_trace()
	
	# placeholder for input variables
	x_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE,)+train_x.shape[1:])
	y_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
	
	# get the loss function and the prediction function for the network
	pred_op = lstm_model_fn(x_placeholder, vocab_size, embed_length)
	loss_op = tf.losses.sparse_softmax_cross_entropy(labels=y_placeholder, logits=pred_op)
	
	# define optimizer
	optimizer = tf.train.AdamOptimizer(LR)
	train_op = optimizer.minimize(loss_op)
	# start tensorflow session
	sess = tf.Session()
	
	# initialization
	init = tf.global_variables_initializer()
	sess.run(init)

    # train loop -----------------------------------------------------
	for epoch in range(NUM_EPOCHS):
		running_loss = 0.0
		n_batch = 0
		for i in range(0, train_x.shape[0]-BATCH_SIZE, BATCH_SIZE):
			# get batch data
			x_batch = train_x[i:i+BATCH_SIZE]
			y_batch = train_y[i:i+BATCH_SIZE]
			# import pdb; pdb.set_trace()
			# run step of gradient descent
			feed_dict = {
				x_placeholder: x_batch,
				y_placeholder: y_batch,
			}
			_, loss_value = sess.run([train_op, loss_op],
									feed_dict=feed_dict)
	
			running_loss += loss_value
			n_batch += 1
	
		print('[Epoch: %d] loss: %.3f' %
			(epoch + 1, running_loss / (n_batch)))
	
    # test loop -----------------------------------------------------
	all_predictions = np.zeros((0, 1))
	for i in range(0, test_x.shape[0], BATCH_SIZE):
		x_batch = test_x[i:i+BATCH_SIZE]
	
		padded = BATCH_SIZE - x_batch.shape[0]
		if padded > 0:
			x_batch = np.pad(x_batch,
							((0, padded), (0, 0)),
							'constant')
	
		feed_dict = {x_placeholder: x_batch}
		batch_pred = sess.run(pred_op,
							feed_dict=feed_dict)
	
		if padded > 0:
			batch_pred = batch_pred[0:-padded]
	
		batch_pred = np.argmax(batch_pred, axis=1)
	
		all_predictions = np.append(all_predictions, batch_pred)
	
	print("Accuracy: %f" % np.mean(all_predictions == test_y))
	plot_confusion_matrix(all_predictions, test_y, title="LSTM")
	# plt.show()
	

if __name__ == "__main__":
    main()
