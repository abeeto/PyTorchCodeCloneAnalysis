import tensorflow as tf 

(_, _), (x_test, y_test) =  tf.keras.datasets.cifar10.load_data()

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

show_multiple_images(x_test, y_test)

test_images = np.array([i for i, j in zip(x_test, y_test) if j in list(map.keys())])
test_labels = np.array([j for i, j in zip(x_test, y_test) if j in list(map.keys())])

show_multiple_images(test_images, test_labels)

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
        
net = Net()

net.load_weights("cifar_two_model")

predictions = net(test_images, training=False)

test_labels = np.where(test_labels == 2, 1, test_labels)

correct = 0
total = 0
for index, pred in enumerate(predictions):
    if tf.math.argmax(pred).numpy().item() == test_labels[index]:
        correct += 1
    total += 1

print(correct / total * 100)

map = {0:"plane", 1:"bird"}
plt.figure(figsize=(10, 10))
for i in range(25):
    predicted = tf.math.argmax(predictions[i]).numpy().item()
    correct = test_labels[i]
    if predicted == correct:
        color = "green"
    else:
        color='red'
    plt.subplot(5, 5, i + 1)
    plt.imshow(test_images[i])
    title = plt.title(f"Predicted: {map[predicted]}")
    xlabel = plt.xlabel(f"Correct : {map[correct]}")
    plt.setp(title, color=color)
    plt.setp(xlabel, color=color)
plt.tight_layout()
plt.show()
