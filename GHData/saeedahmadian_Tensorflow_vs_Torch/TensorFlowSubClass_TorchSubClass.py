import tensorflow as tf
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import pydot
from torch.utils.tensorboard import SummaryWriter
import os
import shutil



def return_x_y():
    raw_train_data = pd.read_csv('train.csv', header=0)

    encoder = OrdinalEncoder()
    scale = MinMaxScaler(feature_range=(0, 1))

    def cleaning_pipeline(dataframe, encoder):
        new_data = dataframe[dataframe.notna().all(axis=1)]
        obj_cols = new_data.columns[new_data.dtypes == object]
        for name in obj_cols:
            new_data[name] = encoder.fit_transform(new_data[name].values.reshape(-1, 1))
        return new_data

    def process_train_data(dataframe, scale, out_name, portion=2):
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
        x = scale.fit_transform(x)
        y = df[:, -1]
        return x, y

    def upsample_train_data(dataframe, scale, out_name, portion=1):
        neg_samples = dataframe[dataframe[out_name] == 0]
        pos_samples = dataframe[dataframe[out_name] == 1]
        N_sample = portion * neg_samples.shape[0]
        pos_resamples = resample(pos_samples,
                                 replace=True,  # sample with replacement
                                 n_samples=N_sample,  # to match majority class
                                 random_state=123)

        df = pd.concat([neg_samples, pos_resamples], axis=0).values
        np.random.shuffle(df)
        x = df[:, 0:-1]
        x = scale.fit_transform(x)
        y = df[:, -1]
        return x, y

    x_train, y_train = upsample_train_data(cleaning_pipeline(raw_train_data, encoder),
                                           scale, raw_train_data.columns[-1], 1)

    return x_train,y_train


class tf_classifier(tf.keras.Model):
    def __init__(self,input_feature,output_feature):
        super(tf_classifier,self).__init__()
        self.in_layer= tf.keras.layers.Dense(units=20,input_shape=(input_feature,1),name='tf_input_layer')
        self.hidd1 = tf.keras.layers.Dense(units=10,activation=tf.nn.relu,name='tf_hidd_layer1')
        self.hidd2 = tf.keras.layers.Dense(units=10,activation=tf.nn.relu,name='tf_hidd_layer2')
        self.concat= tf.keras.layers.Concatenate(axis=-1,name='tf_concatenate_layer')
        self.out = tf.keras.layers.Dense(units=output_feature,name='tf_raw_out_layer')
        if output_feature==1:
            self.out_= tf.keras.layers.Activation(tf.nn.sigmoid,name='tf_sigmoid_layer')
        else:
            self.out_ = tf.keras.layers.Activation(tf.nn.softmax,name='tf_softmax_layer')
    @tf.function
    def call(self,x_in):
        x_in= tf.convert_to_tensor(x_in,dtype=tf.float32,name='convert_tensor')
        x= self.in_layer(x_in)
        x1= self.hidd1(x)
        x2= self.hidd2(x)
        x_out = self.concat([x1,x2])
        x_out = self.out(x_out)
        return self.out_(x_out)


class torch_classifier(torch.nn.Module):
    def __init__(self,input_feature,output_feature):
        super(torch_classifier,self).__init__()
        self.in_layer= torch.nn.Linear(in_features=input_feature,out_features=20,bias=True)
        self.in_layer_act = torch.nn.ReLU()
        self.hidd1 = torch.nn.Linear(20,10,True)
        self.hidd2 = torch.nn.Sequential(torch.nn.Linear(20, 10, True), torch.nn.ReLU())
        self.out = torch.nn.Linear(20,output_feature)
        if output_feature==1:
            self.out_ = torch.nn.Sigmoid()
        else:
            self.out_ = torch.nn.Softmax(dim=-1)

    def forward(self,x):
        x=self.in_layer(x)
        x=self.in_layer_act(x)
        x1= torch.nn.functional.relu(self.hidd1(x))
        x2= self.hidd2(x)
        x= self.out(torch.cat([x1,x2],dim=-1))
        return self.out_(x)


x_train,y_train= return_x_y()

def clean_logs(logdir):
    for file in os.listdir(logdir):
        try:
            os.remove(logdir+'\\'+file)
        except:
            shutil.rmtree(logdir+'\\'+file)

torch_model= torch_classifier(x_train.shape[1],1)
clean_logs('logs_torch')
torch_writer= SummaryWriter(log_dir='logs_torch')
torch_writer.add_graph(torch_model,torch.FloatTensor(x_train))
torch_optim= torch.optim.Adam(torch_model.parameters(),lr=.005)
torch_loss= torch.nn.BCELoss()

def torch_train(x,y,epochs,batch_size):
    batch_num = int(x.shape[0] / batch_size)
    step=0
    df= np.concatenate([x,y.reshape((-1,1))],axis=-1)
    for epoch in range(epochs):
        perm= np.random.permutation(df.shape[0])
        df_shuffled= df[perm,:]
        x_train= torch.FloatTensor(df_shuffled[:,0:-1])
        y_train = torch.FloatTensor(df_shuffled[:,-1])
        for batch in range(batch_num):
            x_batch = x_train[batch * batch_size:(batch + 1) * batch_size, :]
            y_batch = y_train[batch * batch_size:(batch + 1) * batch_size]
            torch_optim.zero_grad()
            loss= torch_loss(torch_model.forward(x_batch),y_batch)
            print('epoch {} -- step {} -- loss {} '.format(epoch, batch, loss))
            torch_writer.add_scalar('Loss_value',loss,step)
            torch_writer.add_histogram('y_out',torch_model.forward(x_batch),step)
            loss.backward()
            torch_optim.step()
            i=0
            for param in torch_model.parameters():
                torch_writer.add_histogram('weight_'+str(i), param.data, step)
                torch_writer.add_histogram('grad_weight_'+str(i), param.grad.data, step)
                i+=1

            step+=1


torch_train(x_train,y_train,10,100)


# consider the classifiers multi-class

tf_optim= tf.keras.optimizers.Adam(learning_rate=.005)
clean_logs('logs_tf')
writer = tf.summary.create_file_writer('./logs_tf')
tf_model= tf_classifier(9,2)

## show graph
tf.summary.trace_on(True,False)
y_out = tf_model(x_train)
with writer.as_default():
    tf.summary.trace_export('My_Simple_Graph',0,'logs_tf')

# writer.set_as_default()
def tf_train(x,y,epochs,batch_size):
    batch_num = int(x.shape[0] / batch_size)
    step=0
    for epoch in range(epochs):
        df = tf.random.shuffle(tf.concat([x, y], axis=1))
        x_train = df[:, 0:-1]
        # y_train= tf.expand_dims(df[:,-1],-1)
        y_train = df[:, -1]
        for batch in range(batch_num):
            x_batch = x_train[batch * batch_size:(batch + 1) * batch_size, :]
            y_batch = y_train[batch * batch_size:(batch + 1) * batch_size]
            with tf.GradientTape() as t:
                y_out = tf_model.call(tf.cast(x_batch,tf.float32))
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_batch,y_out))
            grads = t.gradient(loss, tf_model.trainable_weights)
            tf_optim.apply_gradients(zip(grads, tf_model.trainable_weights))
            print('epoch {} -- step {} -- loss {} '.format(epoch, batch, loss))
            with writer.as_default():
                tf.summary.scalar('loss_value', loss, step)
                i = 0
                for param, grad in zip(tf_model.trainable_weights, grads):
                    tf.summary.histogram('weights_' + str(i), param, step)
                    tf.summary.histogram('grad_' + str(i), grad, step)
                    i += 1

            step += 1






tf_train(x_train,y_train.reshape((-1,1)),1,500)

a=1