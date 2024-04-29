import numpy as np
import tensorflow as tf
from dataset import cifar100
from util import plot_confusion_matrix
import matplotlib.pyplot as plt

tf.random.set_random_seed(1212)

NUM_EPOCHS = 15
BATCH_SIZE = 64
LR = 0.01


def cnn_model_fn(festures):
    """
    Redo the 3-layer cnn from lab 4.
    """
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=festures,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Logits Layer
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 32])
    dense = tf.layers .dense(inputs=pool2_flat, units=4, activation=tf.nn.relu)

    return dense


def main():
    # Load dataset
    train_data, test_data = cifar100(1234)
    train_x, train_y = train_data
    test_x, test_y = train_data

    # placeholder for input variables
    x_placeholder = tf.placeholder(tf.float32,
                                   shape=(BATCH_SIZE,) + train_x.shape[1:])
    y_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

    # get the loss function and the prediction function for the network
    pred_op = cnn_model_fn(x_placeholder)
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=y_placeholder,
                                                     logits=pred_op)

    # define optimizer
    optimizer = tf.train.GradientDescentOptimizer(LR)
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

        # pad small batch
        padded = BATCH_SIZE - x_batch.shape[0]
        if padded > 0:
            x_batch = np.pad(x_batch,
                             ((0, padded), (0, 0), (0, 0), (0, 0)),
                             'constant')

        # run step
        feed_dict = {x_placeholder: x_batch}
        batch_pred = sess.run(pred_op,
                              feed_dict=feed_dict)

        # recover if padding
        if padded > 0:
            batch_pred = batch_pred[0:-padded]

        # get argmax to get class prediction
        batch_pred = np.argmax(batch_pred, axis=1)

        all_predictions = np.append(all_predictions, batch_pred)

    print("Accuracy: %f" % np.mean(all_predictions == test_y))
    plot_confusion_matrix(all_predictions, test_y, title="ConvNet")
    plt.show()


if __name__ == "__main__":
    main()
