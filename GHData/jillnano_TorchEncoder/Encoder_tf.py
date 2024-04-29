# !usr/bin/python
# coding=utf-8

import os
import base64
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from tensorflow.python.platform import gfile
from pandas.io.json import json_normalize

from MongoUtils import MongoUtils
from Constants import Order

def load_data():
	return load_data_mongo()
	csv_data = pd.read_csv('data.csv', index_col = 0)
	filenameList = csv_data.filename
	csv_data = csv_data.drop(['filename'], axis = 1)
	# csv_data = csv_data.drop(['chroma_stft'], axis = 1)
	# csv_data = csv_data.drop(['spec_cent'], axis = 1)
	# csv_data = csv_data.drop(['spec_bw'], axis = 1)
	# csv_data = csv_data.drop(['rolloff'], axis = 1)
	# csv_data = csv_data.drop(['zcr'], axis = 1)
	# for i in range(20):
	# 	csv_data = csv_data.drop(['mfcc_%s'%i], axis = 1)
	# scaler = StandardScaler()
	scaler = MinMaxScaler()
	total_X = scaler.fit_transform(np.array(csv_data, dtype = float))
	return total_X, filenameList, 0.018

def load_data_mongo():
	order = deepcopy(Order)
	order[0] = 'title'
	mongo = MongoUtils()
	mongo_data = json_normalize(mongo.findAllMusic())[order]
	filenameList = mongo_data.title
	mongo_data = mongo_data.drop(['title'], axis = 1)
	scaler = MinMaxScaler()
	total_X = scaler.fit_transform(np.array(mongo_data, dtype = float))
	return total_X, filenameList, 0.01

def load_data_android():
	content = json.load(open('data.json', 'r'))
	filenameList = list(content.keys())
	filenameList.sort()
	total = []
	for i in filenameList:
		total.append(content[i])
	total = np.array(total)
	scaler = MinMaxScaler()
	total_X = scaler.fit_transform(total)
	return total_X, filenameList, 0.0285

def load_data_fft():
	csv_data = pd.read_csv('data_peak.csv', index_col = 0)
	filenameList = csv_data.filename
	csv_data = csv_data.drop(['filename'], axis = 1)
	# scaler = StandardScaler()
	scaler = MinMaxScaler()
	total_X = scaler.fit_transform(np.array(csv_data, dtype = float))
	return total_X, filenameList, 0.0006

def train():

	total_X, _, threshold = load_data()
	x_train, x_test, y_train, y_test = train_test_split(total_X, total_X, test_size = 0.2)
	print(type(x_train))

	learning_rate = 0.01
	training_epochs = 5000
	# batch_size = 64
	display_step = 100
	# examples_to_show = 10
	n_input = total_X.shape[1]
	 
	# tf Graph input (only pictures)
	X = tf.placeholder('float', [None, n_input], name = 'input')

	# 权重和偏置的变化在编码层和解码层顺序是相逆的
	# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数
	weights = {
		'encoder_h1': tf.Variable(tf.random_normal([n_input, 16])),
		'encoder_h2': tf.Variable(tf.random_normal([16, 8])),
		'encoder_h3': tf.Variable(tf.random_normal([8, 2])),

		'decoder_h1': tf.Variable(tf.random_normal([2, 8])),
		'decoder_h2': tf.Variable(tf.random_normal([8, 16])),
		'decoder_h3': tf.Variable(tf.random_normal([16, n_input])),
	}
	biases = {
		'encoder_b1': tf.Variable(tf.random_normal([16])),
		'encoder_b2': tf.Variable(tf.random_normal([8])),
		'encoder_b3': tf.Variable(tf.random_normal([2])),

		'decoder_b1': tf.Variable(tf.random_normal([8])),
		'decoder_b2': tf.Variable(tf.random_normal([16])),
		'decoder_b3': tf.Variable(tf.random_normal([n_input])),
	}

	# 每一层结构都是 xW + b
	# 构建编码器
	def encoder(x):
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
		layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']), name = 'encoder')
		return layer_3

	# 构建解码器
	def decoder(x):
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
		layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']), name = 'decoder')
		return layer_3

	# 构建模型
	encoder_op = encoder(X)
	decoder_op = decoder(encoder_op)

	# 预测
	y_pred = decoder_op
	y_true = X

	# 定义代价函数和优化器
	cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) #最小二乘法
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	saver = tf.compat.v1.train.Saver()
	with tf.Session() as sess:
		# tf.initialize_all_variables() no long valid from
		# 2017-03-02 if using tensorflow >= 0.12
		if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
			init = tf.initialize_all_variables()
		else:
			init = tf.global_variables_initializer()
		sess.run(init)
		# 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
		# total_batch = int(x_train.shape[0]/batch_size) #总批数
		for epoch in range(training_epochs):
			# for i in range(total_batch):
			# batch_xs, batch_ys = x_train[i: i + batch_size].astype(np.float32), y_train[i: i + batch_size].astype(np.float32)
			# batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([optimizer, cost], feed_dict = {X: total_X})
			if epoch % display_step == 0:
				print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(c))
			if c < threshold:
				break
		if c < threshold:
			saver.save(sess, 'Model/model.ckpt')
			print('Optimization Finished!')
		else:
			print('Optimization Failed!')

def predict():
	total_X, filenameList, _ = load_data()

	with tf.Session() as sess:
		saver = tf.compat.v1.train.import_meta_graph('Model/model.ckpt.meta')
		saver.restore(sess, 'Model/model.ckpt')
		# print([tensor.name for tensor in tf.get_default_graph().as_graph_def().node])
		# print(tf.get_default_graph().get_tensor_by_name('encoder_h1'))
		X = tf.get_default_graph().get_tensor_by_name('input:0')
		encoder_op = tf.get_default_graph().get_tensor_by_name('encoder:0')
		result = sess.run(encoder_op, feed_dict={X: total_X})
		# print(ret.shape)
	return result, filenameList

def predict_pb():
	total_X, filenameList, _ = load_data()
	with open('torch_model.pb', 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	graph = tf.Graph()
	with tf.Session(graph = graph) as sess:
	# with graph.as_default():
		tf.import_graph_def(graph_def, name='torch')
		X = tf.get_default_graph().get_tensor_by_name('torch/input:0')
		encoder_op = tf.get_default_graph().get_tensor_by_name('torch/encoder:0')
		result = sess.run(encoder_op, feed_dict={X: total_X})
	return result, filenameList

def predict_data(total_X):
	with open('torch_model.pb', 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	graph = tf.Graph()
	with tf.Session(graph = graph) as sess:
	# with graph.as_default():
		tf.import_graph_def(graph_def, name='torch')
		X = tf.get_default_graph().get_tensor_by_name('torch/input:0')
		encoder_op = tf.get_default_graph().get_tensor_by_name('torch/encoder:0')
		result = sess.run(encoder_op, feed_dict={X: total_X})
	return result

def test(idx):
	# ret, filenameList = predict_pb()
	# ret, filenameList, _ = load_data_mongo()
	ret, filenameList = predict()
	print(filenameList[idx])
	dev = ret[idx]
	result = []
	count = 0
	# for i in csv_data.iterrows():
	for i in ret:
		devb = i
		dist2 = np.sqrt(np.sum(np.square(dev - devb)))
		result.append((filenameList[count], dist2))
		count += 1
	result = sorted(result, key = lambda x: x[1])[0:100]
	for k, v in enumerate(result):
		print(k, *v)

def freeze():
	with tf.Session(graph=tf.get_default_graph()) as sess:
		saver = tf.compat.v1.train.import_meta_graph('Model/model.ckpt.meta')
		input_graph_def = sess.graph.as_graph_def()
		saver.restore(sess, 'Model/model.ckpt')
		X = tf.get_default_graph().get_tensor_by_name('input:0')
		# encoder_op = tf.get_default_graph().get_tensor_by_name('encoder:0')
		output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
																		input_graph_def,
																		['encoder'])
		# 4. 写入文件
		with open('torch_model.pb', 'wb') as f:
			f.write(output_graph_def.SerializeToString())

def preview():
	data, filenameList = predict()
	data = data.T
	X = data[0] * 255
	Y = data[1] * 255
	for i in range(len(X)):
		c = '#9b%s%s'%(hex(int(X[i])).split('x')[-1].zfill(2), hex(int(Y[i])).split('x')[-1].zfill(2))
		plt.scatter(X[i], Y[i], s = 5, c = c)
	plt.savefig('preview.jpg')

def saveToSqlite():
	import Utils
	ret, filenameList = predict_pb()
	ret = ret.T
	X = ret[0] * 255
	Y = ret[1] * 255
	dataList = []
	f = open('peak_data.json', 'r')
	for i in f:
		dataList.append(json.loads(i))
	def createStr(column):
		column = [' '.join(i) for i in column]
		column = ', '.join(column)
		return column
	def insertStr(column, data):
		keys = []
		values = []
		for i in column:
			keys.append(i[0])
			if i[0] == '`_id`':
				values.append('NULL')
			elif i[0] == '`filename`':
				values.append("'%s'"%Utils.getSqlFilename(data['filename']))
			else:
				k = i[0].strip('`')
				values.append("'%s'"%data[k])
		return ', '.join(keys), ', '.join(values)
	SourceColumn = [
		('`_id`', 'INTEGER primary key'),
		('`filename`', 'TEXT'),
		('`unique_id`', 'TEXT UNIQUE'),
		('`mean_0`', 'FLOAT'),
		('`std_0`', 'FLOAT'),
		('`max_0`', 'FLOAT'),
		('`min_0`', 'FLOAT'),
		('`mean_1`', 'FLOAT'),
		('`std_1`', 'FLOAT'),
		('`max_1`', 'FLOAT'),
		('`min_1`', 'FLOAT'),
		('`mean_2`', 'FLOAT'),
		('`std_2`', 'FLOAT'),
		('`max_2`', 'FLOAT'),
		('`min_2`', 'FLOAT')
	]
	EncodeColumn = [
		('`_id`', 'INTEGER primary key'),
		('`filename`', 'TEXT'),
		('`unique_id`', 'TEXT UNIQUE'),
		('`encode_0`', 'FLOAT'),
		('`encode_1`', 'FLOAT')
	]
	sqliteConn = sqlite3.connect('torch_peak')
	sqliteCursor = sqliteConn.cursor()
	sqliteCursor.execute('CREATE TABLE torch_source (%s)'%createStr(SourceColumn))
	# sqliteCursor.execute('CREATE INDEX INDEXID on torch_source(_id)')
	sqliteCursor.execute('CREATE INDEX FILENAME on torch_source(filename)')
	sqliteCursor.execute('CREATE INDEX UNIQUE_ID on torch_source(unique_id)')
	sqliteCursor.execute('CREATE TABLE torch_encode (%s)'%createStr(EncodeColumn))
	sqliteCursor.execute('CREATE INDEX FILENAMEENCODE on torch_encode(filename)')
	sqliteCursor.execute('CREATE INDEX UNIQUE_ID_ENCODE on torch_encode(unique_id)')
	for idx in range(len(dataList)):
		try:
			source_data = dataList[idx]
			if 'unique_id' not in source_data:
				source_data['unique_id'] = Utils.unique_id(filename)
			keys, values = insertStr(SourceColumn, source_data)
			sqliteCursor.execute('INSERT INTO torch_source (%s) VALUES (%s)'%(keys, values))
			data = {'filename': source_data['filename'], 'unique_id': source_data['unique_id'], 'encode_0': X[idx], 'encode_1': Y[idx]}
			keys, values = insertStr(EncodeColumn, data)
			sqliteCursor.execute('INSERT INTO torch_encode (%s) VALUES (%s)'%(keys, values))
		except sqlite3.IntegrityError:
			print(data['filename'], values)
	sqliteConn.commit()
	sqliteCursor.close()

def test_temp():
	import Utils
	f = open('peak_data.json', 'r')
	for i in f:
		filename = json.loads(i)['filename']
		filename = os.path.basename(filename)
		fn = os.path.splitext(filename)[0]
		name = Utils.formatFilename(fn)
		print(name)

if __name__ == '__main__':
	# saveToSqlite()
	# test_temp()
	# train()
	test(122)
	print('=' * 100)
	test(81)
	print('=' * 100)
	test(39)
	# preview()
	# freeze()
	# load_data_android()