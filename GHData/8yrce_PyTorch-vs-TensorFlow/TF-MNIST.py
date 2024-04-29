"""
TF-MNIST.py
    MNIST model using TensorFlow
Bryce Harrington
09/16/22
"""

# TF / Keras imports
import keras.callbacks
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import sparse_categorical_accuracy
from keras.datasets import fashion_mnist

# our config file imports
from Config import LEARNING_RATE, BATCH_SIZE, EPOCHS, VAL_SPLIT

# load our test, train data
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# generate the model architecture
model = Sequential()
model.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(5, 5), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=SparseCategoricalCrossentropy(),
    metrics=[sparse_categorical_accuracy]
)

# define callbacks
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint("./TF_MODEL.h5")

# train and define model hyperparams ( including setting an auto validation split )
history = model.fit(train_data,
                    train_labels,
                    validation_split=VAL_SPLIT,
                    callbacks=[early_stopping, model_checkpoint],
                    epochs=EPOCHS, batch_size=BATCH_SIZE)

# evaluate the model
output = model.evaluate(test_data)
print(output)
