# -*- coding: utf-8 -*-
"""stage2_gan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RPC3zsLnNZsUvlY2oVKBSCvg8AxSn1Tw
"""

import numpy as np
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge
from keras.layers import Reshape,LeakyReLU,ZeroPadding2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.core import Flatten
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, Adagrad
from PIL import Image
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.objectives import binary_crossentropy
import tensorflow as tf
from tqdm import tqdm
import scipy.misc as im

channels = 3

def convolution(inputs,filters,step,stride=2,Normal=True):
    encoder = ZeroPadding2D(padding=(1,1))(inputs)
    encoder = Convolution2D(filters,4,4,subsample=(stride,stride),name='conv_%d'%step)(encoder)
    if Normal:
        encoder = BatchNormalization(name='CBat_%d'%step)(encoder)
    encoder = LeakyReLU(alpha=0.2,name='CLRelu_%d'%step)(encoder)
    return encoder

def deconvolution(inputs,filters,step,dropout):
    _,height,width,_ = (inputs.get_shape()).as_list()
    decoder = Deconvolution2D(filters,4,4,
                              output_shape=(None,2*height,2*width,filters),
                              subsample=(2,2),
                              border_mode='same',
                              name='Deconv_%d' % (8-step))(inputs)
    decoder = BatchNormalization(name='DBat_%d' % (8-step))(decoder)
    if step == 8:
        decoder = Activation(activation='tanh')(decoder)
    else:
        decoder = LeakyReLU(alpha=0.2,name='DLRelu_%d' % (8-step))(decoder)   
    if dropout[step-1] > 0:
        decoder = Dropout(dropout[step-1])(decoder)
    return decoder

def generator_model():
    # Dimensions of image
    img_x = 512
    img_y = 512
    g_inputs = Input(shape=(img_x,img_y,3))
    encoder_filter = [64,128,256,512,512,512,512]
    Encoder = []

    nb_layer = len(encoder_filter)

    decoder_filter = encoder_filter[::-1]
    dropout = [0.5,0.5,0.5,0,0,0,0,0]

    for i in range(nb_layer):
        if i == 0:
            encoder = convolution(g_inputs,encoder_filter[i],i+1)
        else:
            encoder = convolution(encoder,encoder_filter[i],i+1)
        Encoder.append(encoder)     
        
    #Middle layer...
    middle = convolution(Encoder[-1],512,8)
    
    #Buliding decoder layers...
    for j in range(nb_layer):
        if j == 0:
            decoder = deconvolution(middle,decoder_filter[j],j+1,dropout)
        else:
            decoder = merge([decoder,Encoder[nb_layer-j]],mode='concat',concat_axis=-1)
            decoder = deconvolution(decoder,decoder_filter[j],j+1,dropout)
            
    #Generate original size's originals
    g_output = merge([decoder,Encoder[0]],mode='concat',concat_axis=-1)
    g_output = deconvolution(g_output,3,8,dropout)
    
    model = Model(g_inputs,g_output)
    return model

def discriminator_model():
    inputs = Input(shape=(img_cols,img_rows,channels*2))
    d = ZeroPadding2D(padding=(1,1))(inputs)
    d = Convolution2D(64,4,4,subsample=(2,2))(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = ZeroPadding2D(padding=(1,1))(d)
    d = Convolution2D(128,4,4,subsample=(2,2))(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = ZeroPadding2D(padding=(1,1))(d)
    d = Convolution2D(256,4,4,subsample=(2,2))(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = ZeroPadding2D(padding=(1,1))(d)
    d = Convolution2D(512,4,4,subsample=(1,1))(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = ZeroPadding2D(padding=(1,1))(d)
    # Sigmoid actiation 
    d = Convolution2D(1,4,4,subsample=(1,1),activation='sigmoid')(d)
    model = Model(inputs,d)
    return model

def generator_containing_discriminator(generator, discriminator):
    inputs = Input((img_cols, img_rows,channels))
    x_generator = generator(inputs)
    
    merged = merge([inputs, x_generator], mode='concat',concat_axis=-1)
    discriminator.trainable = False
    x_discriminator = discriminator(merged)
    
    model = Model(inputs,[x_generator,x_discriminator])
    
    return model

def generate_original(generator,output,e):
    original = generator.predict(output)
    original = np.squeeze(original,axis=0)
    output = np.squeeze(output,axis=0)
    im.imsave('output_%d.png' % e,output)
    im.imsave('original_%d.png' % e,original)

np.mean

def discriminator_on_generator_loss(y_true,y_pred):
    # Cross entropy loss function used
    return K.mean(K.binary_crossentropy(y_pred,y_true), axis=(1,2,3))

def generator_l1_loss(y_true,y_pred):
    # Loss is caclulated by computing the difference between true and pred images
    return K.mean(K.abs(y_pred - y_true),axis=(1,2,3))

def train(epochs,batchsize):
    # Loads images from .npy files
    original = np.load('original.npy')
    output = np.load('output.npy')
    original = original.astype('float32')
    output = output.astype('float32')
    # Processes image as [0,1]
    original = (original - 127.5) / 127.5
    output = (output - 127.5) / 127.5
    batchCount = original.shape[0] / batchsize
    print('Epochs',epochs)
    print('Bathc_size',batchsize)
    print('Batches per epoch',batchCount)
    generator = generator_model()
    discriminator = discriminator_model()
    gan = generator_containing_discriminator(generator,discriminator)
    generator.compile(loss=generator_l1_loss, optimizer='RMSprop')
    gan.compile(loss=[generator_l1_loss,discriminator_on_generator_loss] , optimizer='RMSprop')
    discriminator.trainable = True
    discriminator.compile(loss=discriminator_on_generator_loss, optimizer='RMSprop')
    G_loss = []
    D_loss = []
    for e in xrange(1,epochs+1):
        print('-'*15 , 'Epoch %d' % e , '-'*15)
        for _ in tqdm(xrange(batchCount)):
            random_number = np.random.randint(1,original.shape[0],size=batchsize)
            batch_original = original[random_number]
            batch_output = output[random_number]
            batch_output2 = np.tile(batch_output,(2,1,1,1))
            y_dis = np.zeros((2*batchsize,30,30,1))
            y_dis[:batchsize] = 1.0
            generated_original = generator.predict(batch_output)
            # Default is concat first dimension
            concat_original = np.concatenate((batch_original,generated_original))
            
            dis_input = np.concatenate((concat_original,batch_output2),axis=-1)
            dloss = discriminator.train_on_batch(dis_input,y_dis)
            random_number = np.random.randint(1,original.shape[0],size=batchsize)
            train_output = output[random_number]
            batch_original = original[random_number]
            y_gener = np.ones((batchsize,30,30,1))
            discriminator.trainable = False
            gloss = gan.train_on_batch(train_output,[batch_original,y_gener])
            discriminator.trainable = True
        G_loss.append(gloss)
        D_loss.append(dloss)
        if e % 50 == 0 or e == 1:
            generate_original(generator,output[0:1],e)
            # Saves weights in h5 file
            generator.save('Model_para/pix2pix_g_epoch_%d.h5' % e)
            discriminator.save('Model_para/pix2pix_d_epoch_%d.h5' % e)
            gan.save('Model_para/pix2pix_gan_epoch_%d.h5' % e)
    D_loss = np.array(D_loss)
    G_loss = np.array(G_loss)
    np.save('Model_para/dloss.npy',D_loss)
    np.save('Model_para/gloss.npy',G_loss)

# hyperparameters:
epochs = 100
bz = 32
train(epochs,bz)
g = generator_model()
d = discriminator_model()
gan = generator_containing_discriminator(g,d)

