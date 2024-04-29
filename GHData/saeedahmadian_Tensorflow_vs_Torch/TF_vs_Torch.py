import tensorflow as tf
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt


raw_train_data = pd.read_csv('train.csv',header=0)

encoder = OrdinalEncoder()
scale= MinMaxScaler(feature_range=(0,1))
def cleaning_pipeline(dataframe,encoder):
    new_data = dataframe[dataframe.notna().all(axis=1)]
    obj_cols= new_data.columns[new_data.dtypes==object]
    for name in obj_cols:
        new_data[name] = encoder.fit_transform(new_data[name].values.reshape(-1, 1))
    return new_data

def process_train_data(dataframe,scale,out_name,portion=2):
    neg_num = dataframe[dataframe[out_name] == 0].shape[0]
    pos_num = dataframe[dataframe[out_name] == 1].shape[0]
    pos_out = dataframe[dataframe[out_name] == 1]
    if portion != None:
        neg_out = dataframe[dataframe[out_name] == 0].sample(pos_num * portion, random_state=1)
    else:
        neg_out = dataframe[dataframe[out_name] == 0]

    df = pd.concat([neg_out, pos_out], axis=0).values
    np.random.shuffle(df)
    x = df[:, 0:-1]
    x= scale.fit_transform(x)
    y = df[:, -1]
    return x,y

def upsample_train_data(dataframe,scale,out_name,portion=1):
    neg_samples = dataframe[dataframe[out_name] == 0]
    pos_samples = dataframe[dataframe[out_name] == 1]
    N_sample= portion*neg_samples.shape[0]
    pos_resamples = resample(pos_samples,
                                 replace=True,     # sample with replacement
                                 n_samples=N_sample,    # to match majority class
                                 random_state=123)

    df = pd.concat([neg_samples, pos_resamples], axis=0).values
    np.random.shuffle(df)
    x = df[:, 0:-1]
    x= scale.fit_transform(x)
    y = df[:, -1]
    return x,y


def tf_regression(x,y,epochs=1,batch_size=100,reg=True):
    x= tf.convert_to_tensor(x,dtype=tf.float32)
    y = tf.convert_to_tensor(y,dtype=tf.float32)
    y=tf.expand_dims(y,-1)
    num_features= x.shape[1]
    out_features= 1
    loss_list= []
    reg_metric= []

    w= tf.Variable(initial_value=tf.random.truncated_normal(shape=(num_features,out_features),dtype=tf.float32),
                   trainable=True, name='W_tf')
    b = tf.Variable(initial_value=tf.random.truncated_normal(shape=(out_features,),dtype=tf.float32),
                    trainable=True, name='b_tf')
    optim = tf.keras.optimizers.Adam(0.005)
    batch_num= int(x.shape[0]/batch_size)
    for epoch in range(epochs):
        df=tf.random.shuffle(tf.concat([x,y],axis=1))
        x_train= df[:,0:-1]
        y_train= tf.expand_dims(df[:,-1],-1)
        for batch in range(batch_num):
            x_batch= x_train[batch*batch_size:(batch+1)*batch_size,:]
            y_batch = y_train[batch * batch_size:(batch + 1) * batch_size]
            with tf.GradientTape() as t:
                if reg == True:
                    y_out = tf.add(tf.matmul(x_batch, w), b)
                    Loss = tf.reduce_mean(tf.square(y_batch - y_out))
                else:
                    y_out = tf.nn.sigmoid(tf.add(tf.matmul(x_batch, w), b))
                    Loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_batch, logits=y_out))
            print("epoch {} iteration {} loss {}".format(epoch,batch,Loss))
            grads = t.gradient(Loss, [w, b])
            # weights= optim.get_weights()
            # delta_weights= optim.get_gradients(Loss, [w, b])
            optim.apply_gradients(zip(grads, [w, b]))
            acc= tf.keras.metrics.binary_accuracy(y_batch,y_out)

            ## Another way
            # optim.minimize(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_batch, logits=y_out)),[w,b])

            ## We can also update weights without optim class
            ## dw,db= grads
            ##lr=0.005
            ## w.assign_add(lr*dw)
            ## b.assign_add(lr*db)
        loss_list.append(Loss)
        reg_metric.append(acc)
    return [w,b],loss_list


def tf_multi_label_logistic(x,y,epochs=1,batch_size=100):
    x= tf.convert_to_tensor(x,dtype=tf.float32)
    y = tf.convert_to_tensor(y,dtype=tf.float32)
    y=tf.expand_dims(y,-1)
    num_features= x.shape[1]
    out_features= 2
    loss_list= []
    reg_metric= []

    w= tf.Variable(initial_value=tf.random.truncated_normal(shape=(num_features,out_features),dtype=tf.float32),
                   trainable=True, name='W_tf')
    b = tf.Variable(initial_value=tf.random.truncated_normal(shape=(out_features,),dtype=tf.float32),
                    trainable=True, name='b_tf')
    optim = tf.keras.optimizers.Adam(0.005)
    batch_num= int(x.shape[0]/batch_size)
    for epoch in range(epochs):
        df=tf.random.shuffle(tf.concat([x,y],axis=1))
        x_train= df[:,0:-1]
        # y_train= tf.expand_dims(df[:,-1],-1)
        y_train= df[:,-1]
        for batch in range(batch_num):
            x_batch= x_train[batch*batch_size:(batch+1)*batch_size,:]
            y_batch = y_train[batch * batch_size:(batch + 1) * batch_size]
            with tf.GradientTape() as t:
                y_out = tf.add(tf.matmul(x_batch, w), b)
                # y_out = tf.nn.softmax(y_out,axis=-1)
                Loss =tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.cast(y_batch,tf.int64),y_out))
            print("epoch {} iteration {} loss {}".format(epoch,batch,Loss))
            grads = t.gradient(Loss, [w, b])
            # weights= optim.get_weights()
            # delta_weights= optim.get_gradients(Loss, [w, b])
            optim.apply_gradients(zip(grads, [w, b]))
            # acc= tf.keras.metrics.binary_accuracy(y_batch,y_out)

            ## Another way
            # optim.minimize(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_batch, logits=y_out)),[w,b])

            ## We can also update weights without optim class
            ## dw,db= grads
            ##lr=0.005
            ## w.assign_add(lr*dw)
            ## b.assign_add(lr*db)
        loss_list.append(Loss)
        # reg_metric.append(acc)
    return [w,b],loss_list



def torch_regression(x,y,epochs=1,batch_size=100,reg= True):
    num_features= x.shape[1]
    out_features= 1
    df= np.concatenate([x,y.reshape((-1,1))],axis=1)
    loss_list= []
    reg_metric= []
    w= torch.autograd.Variable(torch.rand([num_features,out_features],requires_grad=False,dtype=torch.float32),
                               requires_grad=True,name='W_torch')

    b = torch.autograd.Variable(torch.rand([out_features], requires_grad=False, dtype=torch.float32),
                                requires_grad=True, name='b_torch')

    optim = torch.optim.Adam([w,b],lr=.005)
    batch_num = int(x.shape[0] / batch_size)
    for epoch in range(epochs):
        permutation = np.random.permutation(df.shape[0])
        shuffled_data = df[permutation, :]
        x_train= torch.FloatTensor(shuffled_data[:,0:-1])
        y_train = torch.FloatTensor(shuffled_data[:,-1].reshape((-1,1)))
        for batch in range(batch_num):
            x_batch= x_train[batch*batch_size:(batch+1)*batch_size,:]
            y_batch = y_train[batch * batch_size:(batch + 1) * batch_size,:]
            optim.zero_grad()
            if reg == True:
                y_out = torch.add(torch.matmul(x_batch, w), b)
                Loss = torch.mean((y_out - y_batch) ** 2)
            else:
                y_out = torch.sigmoid(torch.add(torch.matmul(x_batch, w), b))
                Loss = torch.nn.functional.binary_cross_entropy(y_out,y_batch)

            print("epoch {} iteration {} loss {}".format(epoch, batch, Loss))
            Loss.backward()
            optim.step()
            ## calculate wirghts and grads are easy
            # w.data = w.data -lr*w.grad.data
            # b.data = b.data - lr* b.grad.data
        loss_list.append(Loss.detach().numpy())
        # reg_metric.append(torch.nn.functional.cross_entropy(y_out,y_batch))
    return [w,b],loss_list

def torch_multi_label_logistic(x,y,epochs=1,batch_size=100):
    num_features= x.shape[1]
    out_features= 2
    df= np.concatenate([x,y.reshape((-1,1))],axis=1)
    loss_list= []
    reg_metric= []
    w= torch.autograd.Variable(torch.rand([num_features,out_features],requires_grad=False,dtype=torch.float32),
                               requires_grad=True,name='W_torch')

    b = torch.autograd.Variable(torch.rand([out_features], requires_grad=False, dtype=torch.float32),
                                requires_grad=True, name='b_torch')

    optim = torch.optim.Adam([w,b],lr=.005)
    batch_num = int(x.shape[0] / batch_size)
    for epoch in range(epochs):
        permutation = np.random.permutation(df.shape[0])
        shuffled_data = df[permutation, :]
        x_train= torch.FloatTensor(shuffled_data[:,0:-1])
        y_train = torch.tensor(shuffled_data[:,-1],dtype=torch.long)
        for batch in range(batch_num):
            x_batch= x_train[batch*batch_size:(batch+1)*batch_size,:]
            y_batch = y_train[batch * batch_size:(batch + 1) * batch_size]
            optim.zero_grad()
            y_out = torch.add(torch.matmul(x_batch, w), b)
            y_out = torch.nn.functional.softmax(y_out,-1)
            # y_out = torch.softmax(y_out)
            Loss= torch.nn.functional.cross_entropy(y_out,y_batch)
            print("epoch {} iteration {} loss {}".format(epoch, batch, Loss.detach().numpy()))
            Loss.backward()
            optim.step()
            ## calculate wirghts and grads are easy
            # w.data = w.data -lr*w.grad.data
            # b.data = b.data - lr* b.grad.data
        loss_list.append(Loss.detach().numpy())
        # reg_metric.append(torch.nn.functional.cross_entropy(y_out,y_batch))
    return [w,b],loss_list


x_train,y_train = upsample_train_data(cleaning_pipeline(raw_train_data,encoder),
                                      scale,raw_train_data.columns[-1],1)

# [w,b],loss= tf_regression(x_train,y_train,100,1000,False)
# [w,b],loss= torch_regression(x_train,y_train,100,1000,False)
# [w,b],loss = tf_multi_label_logistic(x_train,y_train,epochs=100,batch_size=1000)
[w,b],loss= torch_multi_label_logistic(x_train,y_train,epochs=100,batch_size=1000)

"""
conclusion:
torch.autograd.Variable() ==> tf.Variable()
torch.nn.function. ...    ==> tf.nn. ...
torch.nn. Seqential, Linear, Softmax ==> tf.keras.layers. Seqential, Dense, Softmax
torch.optim              ==> tf.keras.optimizers
torch.nn.Loss+ MSE, BCE and etc ==> tf.keras.losses. ...
torch.nn.Module ==> tf.keras.Model
Loss.Backward() calculated gradients ==> with tf.GradientTape() as t : Loss and t.gradient() calculated gradients
optim.step    ==> optim.applygradient(grad_var)




"""

a=1















