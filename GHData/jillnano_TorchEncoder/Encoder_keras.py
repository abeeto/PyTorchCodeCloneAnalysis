# !usr/bin/python
# coding=utf-8

import random
from glob import glob
import numpy as np
# from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, Input, UpSampling2D, MaxPooling2D, Dense
from keras.preprocessing.image import load_img, img_to_array
# from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model, Model
from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt

INPUT_SHAPE = (25,)

def make_model():

	input_layout = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format

	x = Dense(16, activation='relu')(input_layout)
	x = Dense(12, activation='relu')(x)
	x = Dense(8, activation='relu')(x)
	x = Dense(4, activation='relu')(x)
	encoded = Dense(2, activation='relu')(x)

	x = Dense(4, activation='relu')(encoded)
	x = Dense(8, activation='relu')(x)
	x = Dense(12, activation='relu')(x)
	x = Dense(16, activation='relu')(x)
	decoded = Dense(25, activation='sigmoid')(x)

	autoencoder = Model(input_layout, decoded)
	encoder = Model(input_layout, encoded)
	# adam = Adam(lr = 1e-4)
	autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
	autoencoder.summary()
	return autoencoder, encoder, input_layout

def build(model_path = None):
	autoencoder, encoder_model, input_layout = make_model()
	if model_path:
		autoencoder = load_model(model_path)
		autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
	# return
	csv_data = pd.read_csv('data.csv', index_col = 0)
	csv_data = csv_data.drop(['filename'], axis = 1)
	# scaler = StandardScaler()
	scaler = MinMaxScaler()
	X = scaler.fit_transform(np.array(csv_data, dtype = float))
	# X = (X - X.min()) / (X.max() - X.min())
	# X = Y = csv_data.as_matrix()
	# x_train, x_test, y_train, y_test = train_test_split(X, X, test_size = 0.2)
	x_train, y_train, x_test, y_test = X, X, X, X

	# modelCheckpoint = ModelCheckpoint('log/ep{epoch:03d}-loss{loss:.4f}.h5', monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
	# earlyStopping = EarlyStopping(monitor='loss', min_delta=0.0005, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=False)

	autoencoder.fit(x_train, y_train,
		verbose = 1,
		epochs = 128,
		# batch_size = 50,
		shuffle = True,
		validation_data = (x_test, y_test))
	autoencoder.save('model.h5')
	encoder_model.save('encoder_model.h5')

def test_model():

	input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

	x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)
	print(encoded.shape)
	# at this point the representation is (4, 4, 8) i.e. 128-dimensional

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(16, (3, 3), activation='relu')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
	print(decoded.shape)

	autoencoder = Model(input_img, decoded)
	encoder = Model(input_img, encoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	autoencoder.summary()
	return autoencoder, encoder, input_img

def test():
	csv_data = pd.read_csv('data.csv', index_col = 0)
	filenameList = csv_data.filename
	csv_data = csv_data.drop(['filename'], axis = 1)
	scaler = MinMaxScaler()
	X = scaler.fit_transform(np.array(csv_data, dtype = float))
	# x_train, x_test, y_train, y_test = train_test_split(X, X, test_size = 0.2)
	encoder = load_model('encoder_model.h5')
	ret = encoder.predict(X)
	print(ret.shape)
	dev = ret[132]
	result = []
	count = 0
	# for i in csv_data.iterrows():
	for i in ret:
		devb = i
		dist2 = np.sqrt(np.sum(np.square(dev - devb)))
		result.append((filenameList[count], dist2))
		count += 1
	result = sorted(result, key = lambda x: x[1])
	print(result)

if __name__ == '__main__':
	build()
	test()
	# test_model()