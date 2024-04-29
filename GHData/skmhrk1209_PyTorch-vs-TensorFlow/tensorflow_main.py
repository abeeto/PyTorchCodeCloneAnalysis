import tensorflow as tf
import numpy as np
import time


tf.logging.set_verbosity(tf.logging.INFO)


def conv_net(features, labels, mode):

    inputs = tf.reshape(
        tensor=features["images"],
        shape=[-1, 28, 28, 1]
    )
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=[2, 2]
    )
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=[2, 2]
    )
    inputs = tf.layers.flatten(inputs)
    inputs = tf.layers.dense(
        inputs=inputs,
        units=1024,
        activation=tf.nn.relu
    )
    logits = tf.layers.dense(
        inputs=inputs,
        units=10
    )
    predictions = tf.argmax(
        input=logits,
        axis=1
    )
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=tf.train.AdamOptimizer().minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
        )
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                accuracy=tf.metrics.accuracy(
                    labels=labels,
                    predictions=predictions
                )
            )
        )


if __name__ == "__main__":

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = np.asarray(mnist.train.images, dtype=np.float32)
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = np.asarray(mnist.test.images, dtype=np.float32)
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=conv_net,
        model_dir="mnist_convnet_model",
        config=tf.estimator.RunConfig(
            save_summary_steps=1000,
            save_checkpoints_steps=1000,
            log_step_count_steps=100
        )
    )
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=10,
        shuffle=True
    )
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": eval_data},
        y=eval_labels,
        batch_size=100,
        num_epochs=1,
        shuffle=False
    )

    begin = time.time()
    mnist_classifier.train(train_input_fn)
    print(mnist_classifier.evaluate(eval_input_fn))
    end = time.time()

    print("elapsed_time: {}s".format(end - begin))
