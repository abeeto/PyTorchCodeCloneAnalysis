from abc import ABC

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, save_model, model_from_json
from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, MaxPooling2D, AveragePooling2D, Flatten, Layer
from tensorflow.keras.layers import BatchNormalization, LSTM
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import activations
from tensorflow.keras import losses

train_set_path = 'dataset/training_set/'
test_set_path = 'dataset/test_set/'

train_cats = os.listdir('dataset/training_set/cats/')
train_dogs = os.listdir('dataset/training_set/dogs/')

test_cats = os.listdir('dataset/test_set/cats/')
test_dogs = os.listdir('dataset/test_set/dogs/')

train_images = []
train_labels = []
test_images = []
test_labels = []

for i in train_cats:
    image = cv2.imread(train_set_path + 'cats/' + i)
    img = cv2.resize(image, (64, 64))
    train_images.append(img)
    train_labels.append(0)

for i in train_dogs:
    image = cv2.imread(train_set_path + 'dogs/' + i)
    img = cv2.resize(image, (64, 64))
    train_images.append(img)
    train_labels.append(1)

for i in test_cats:
    image = cv2.imread(test_set_path + 'cats/' + i)
    img = cv2.resize(image, (64, 64))
    test_images.append(img)
    test_labels.append(0)

for i in test_dogs:
    image = cv2.imread(test_set_path + 'dogs/' + i)
    img = cv2.resize(image, (64, 64))
    test_images.append(img)
    test_labels.append(1)

train_images = np.array(train_images, dtype='float') / 255.0
test_images = np.array(test_images, dtype='float') / 255.0

train_images.shape
train_labels = to_categorical(np.array(train_labels), num_classes=2)
test_labels = to_categorical(np.array(test_labels), num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=0)
X_train.shape


class Classifier(Model):

    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation=activations.relu)
        self.bN = BatchNormalization(axis=-1)
        self.bN2 = BatchNormalization(axis=-1)
        self.pool = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(32, (3, 3), activation=activations.relu)
        self.conv3 = Conv2D(64, (3, 3), activation=activations.relu)
        self.flat = Flatten()
        self.dense1 = Dense(64, activation=activations.relu)
        self.dense2 = Dense(2, activation=activations.softmax)

    def call(self, inputs, training=False, mask=None):
        x = self.conv1(inputs)
        x = self.bN(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bN(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bN2(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.dense1(x)

        return self.dense2(x)

classifier = Classifier()
classifier.compile(loss=losses.categorical_crossentropy, metrics=['accuracy'], optimizer=optimizers.Adam(lr=0.01))
classifier.summary()
classifier.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))