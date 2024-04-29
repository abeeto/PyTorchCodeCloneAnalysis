import tensorflow as tf 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Loading cifar10 dataset
x_train, y_train, x_test, y_test =  np.load("../../train_imgs.npz")['data'], np.load('../../train_lbs.npz')['data'], np.load("../../test_imgs.npz")['data'], np.load("../../test_lbs.npz")['data']

# Printing the shape of the data
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# data map for the images
classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# print out the first 25 images
def show_multiple_images(imgs, labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5,i + 1)
        plt.imshow(imgs[i])
        plt.title(classes[labels[i]])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

show_multiple_images(x_train, y_train)

# filter out data for plane and birds
map = {0:'plane',2:'bird'}

train_images = np.array([i for i, j in zip(x_train, y_train) if j in list(map.keys())])
train_labels = np.array([j for i, j in zip(x_train, y_train) if j in list(map.keys())])


test_images = np.array([i for i, j in zip(x_test, y_test) if j in list(map.keys())])
test_labels = np.array([j for i, j in zip(x_test, y_test) if j in list(map.keys())])

print(f"{len(train_images)} images and {len(train_labels)} labels in our train data")
print(f"{len(test_images)} images and {len(test_labels)} labels in our test data")

show_multiple_images(train_images, train_labels)

train_labels = np.where(train_labels==2, 1, train_labels)

encoded_train_labels = tf.keras.utils.to_categorical(train_labels)
encoded_test_labels = tf.keras.utils.to_categorical(test_labels)


tensor_train_images = tf.data.Dataset.from_tensor_slices(train_images)
tensor_train_labels = tf.data.Dataset.from_tensor_slices(encoded_train_labels)

tensor_train_data = tf.data.Dataset.zip((tensor_train_images, tensor_train_labels))

tensor_train_data = tensor_train_data.batch(16).prefetch(AUTOTUNE)


class Net(tf.keras.Model):
    def __init__(self, input_shape=(None, 32, 32, 3)):
        super(Net, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, 3, input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(8, 3)
        self.pool = tf.keras.layers.MaxPool2D(2)
        self.out = tf.keras.layers.Dense(2)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        for i in ([64, 32]):
            x = tf.keras.layers.Dense(i)(x)
            x = tf.nn.relu(x)
        return self.out(x)
        


optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

net = Net()
epochs = 50
for epoch in range(epochs):
    print("\nStart of epoch %d" %(epoch, ))
    for step, (x_train_batch, y_train_batch) in enumerate(tensor_train_data):
        with tf.GradientTape() as tape:
            logits = net(x_train_batch, training=True)
            loss_value = loss_function(y_train_batch, logits)
        grads = tape.gradient(loss_value, net.trainable_weights)
        optimizer.apply_gradients(zip(grads, net.trainable_weights))
        train_acc_metric.update_state(y_train_batch, logits)
    train_acc = train_acc_metric.result()
    print("Training acc over epoch : %.4f" %(float(train_acc)*100))
    train_acc_metric.reset_states()

net.save_weights("cifar_two_model")


