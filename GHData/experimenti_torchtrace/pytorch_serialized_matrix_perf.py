# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import http.client as http
import harperdb as hdb
import datetime as dt

loss_history = []

now = dt.datetime.now()

def frameToExcel(narray, label):
    df = pd.DataFrame(narray)
    df.to_excel(writer, sheet_name=label)

def frameToHarper(narray, label):
    hdb.insert_narray_serialized(narray, label)

def initSchema():
    hdb.dropSchema()
    hdb.createSchema()
    hdb.createTable('trace')

def runPersistBenchmark(persist='HarperDB'):

	if(persist=='Excel'):
		writer = pd.ExcelWriter('inspection.xlsx', engine='xlsxwriter')

		start_time = dt.datetime.now()
		trainNetwork()
		end_time = dt.datetime.now()

		diff = end_time - start_time 
		elapsed_ms = (diff.days * 86400000) + (diff.seconds * 1000) + (diff.microseconds / 1000)
		print("Training time for: " + persist + " {0}".format(str(elapsed_ms)))

		closeExcel()

	elif(persist=='HarperDB'):

		start_time = dt.datetime.now()
		trainNetwork()
		end_time = dt.datetime.now()

		diff = end_time - start_time 
		elapsed_ms = (diff.days * 86400000) + (diff.seconds * 1000) + (diff.microseconds / 1000)
		print("Training time for: " + persist + " {0}".format(str(elapsed_ms)))
		hdb.exportResults()
		hdb.showLogs()
		hdb.describeSchema()


def trainExpandingNetwork(persist='HarperDB'):

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 1
    # Create random input and output data
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    # Randomly initialize weights
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    learning_rate = 1e-6

    for t in range(500):

        if(t%10 == 0):
            print('epoch' + str(t))

        # Forward pass: compute predicted y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        loss_history.append(loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)
        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

        #Save the weights 
        if(persist=='HarperDB'):

            frameToHarper(w1, 'trace')

def trainNetwork(persist='HarperDB'):

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    # Create random input and output data
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    # Randomly initialize weights
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    learning_rate = 1e-6

    for t in range(500):

        if(t%10 == 0):
            print('epoch' + str(t))

        # Forward pass: compute predicted y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        loss_history.append(loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)
        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

        #Save the weights 
        if(persist=='HarperDB'):
            frameToHarper(w1, 'trace')

initSchema()
runPersistBenchmark()

