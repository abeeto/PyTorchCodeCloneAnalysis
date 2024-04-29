from __future__ import print_function, division
from Batch_Manager import Batcher

from keras import backend as K
from keras.layers import Conv2DTranspose, Conv2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import os
import numpy as np

class DCGAN():
    '''
    A Deep Convolutional Generative Adversarial Network
    
    FloyedHub Generator (https://github.com/ReDeiPirati/dcgan):
    (1,1,100) -> (4,4,512) -> (8,8,256) -> (16,16,128) 
              -> (32,32,64) -> (64,64,3)
    with dropout, batch normalization, and relu (in that order)
    
    Paper Generator (https://arxiv.org/abs/1511.06434):
    (100,) -> (4*4*1024,) -> (4,4,1024)  -> (8x8x512) 
           -> (16x16x256) -> (32x32x128) -> (64,64,3)
    with relu, dropout, and batch normalization (in that order)
    
    Discriminator:
    (64,64,3) -> (32,32,64) -> (16,16,128) -> (8,8,256)
              -> (4,4,512)  -> flat -> Sigmoid
    with LeakyReLU, dropout, and batch norma. (in that order)
    
    !!! IMPORTANT: If you attempt to load weights of the one
                   generator while using the other, the process
                   will fail.
    !!! The Paper Generator provides superior results, however
        the FloyedHub one comes with pre-trained weights on the
        CelebA database. Thus, when setting transfer_learing=True
        bellow the latter is used.
    '''
    def __init__(self, generator='FloyedHub', transfer_learing=False):
        '''
        generator -> ('FloyedHub', 'Paper')
        transfer_learing -> bool, set to True to train only the last
                            layer of the generator
        1) Setup Hyperparameters
        2) Creates directories where the generated images and 
           the weights of the trained model will be saved
        3) Builds and compiles the model 
        4) Loads weights (if found in the corresponding dir)
        '''
        
        #======================
        # Setup Hyperparameters
        #======================
        
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        if generator == 'FloyedHub':
            self.latent_dim = (1,1,100)
        else:
            self.latent_dim = (100,)
        # Generator:
        self.gen_learning_rate = 0.0002
        self.gen_clipvalue     = 0.1
        self.gen_dropout       = 0.2
        self.gen_momentum      = 0.95
        gen_opt = Adam(self.gen_learning_rate, self.gen_clipvalue)
        # Discriminator:
        self.dis_learning_rate = 0.000005
        self.dis_clipvalue     = 0.1
        self.dis_dropout       = 0.2
        self.dis_momentum      = 0.95
        dis_opt = Adam(self.dis_learning_rate, self.dis_clipvalue)
        
        #======================
        # Setup dir and paths
        #======================
        
        # Creates dict with the paths of parent dir, saved models, etc..
        paths = {'parent':os.path.dirname(os.path.realpath(__file__))}
        paths['weights'] = os.path.join(paths['parent'],'weights')
        paths['generated_imgs'] = os.path.join(paths['parent'],'generated_imgs')
        paths['generator_weights'] = os.path.join(paths['weights'],
                                                  'generator.h5')
        paths['discriminator_weights'] = os.path.join(paths['weights'],
                                                      'discriminator.h5')
        self.paths = paths
        
        # Check if relevant subdirectories exists, if not create them
        if not os.path.exists(self.paths['weights']):
            os.makedirs(self.paths['weights'])
        if not os.path.exists(self.paths['generated_imgs']):
            os.makedirs(self.paths['generated_imgs'])
        
        #======================
        # Build/compile model
        #======================
        
        # Build discriminator and compile it
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
             loss='binary_crossentropy',
             optimizer=dis_opt,
             metrics=[soft_binary_accuracy])
        
        # Build the generator and combine it with the
        # discriminator
        if generator == 'FloyedHub':
            self.generator = self.build_floyedhub_generator()
        else:
            self.generator = self.build_paper_generator()
        
        if transfer_learing:
            # Set all layers, apart from the last K, to
            # non-trainable
            i = 0
            K = 2
            for layer in reversed(self.generator.layers):
                if 'conv2d' in layer.name:
                    i += 1
                    if i <= K:
                        layer.trainable = True
                    else:
                        layer.trainable = False
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input 
        # and determines validity
        z = Input(shape=self.latent_dim)
        img = self.generator(z)
        valid = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=gen_opt)
        
        #=========================
        # Load pre-trained weights
        #=========================
        self.load()
        
    def build_paper_generator(self):
        '''
        Build Generator according to the following paper:
            https://arxiv.org/abs/1511.06434
        Also, very similar to the OpenAI implementation.
        '''
        
        def original_gen_layer(input_, filters, kernel_size, strides):
            '''Layer used to build the generator'''
            out = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                  strides=strides, padding='same',
                                  kernel_initializer=RandomNormal(0, 0.02),
                                  activation='relu')(input_)
            out = Dropout(self.gen_dropout)(out)
            out = BatchNormalization(momentum=self.gen_momentum)(out)
            return out
        
        # Build the generator:
        noise = Input(shape=self.latent_dim) # is (100,)
        # becomes 4*4*1024
        flat = Dense(4*4*1024,kernel_initializer=RandomNormal(0, 0.02))(noise)
        conv = Reshape((4,4,1024))(flat)  # becomes 4x4x1024
        conv = gen_layer(conv, 512, 2, 2) # becomes 8x8x512
        conv = gen_layer(conv, 256, 5, 2) # becomes 16x16x256
        conv = gen_layer(conv, 128, 5, 2) # becomes 32x32x128
        # becomes 64x64x3
        conv = Conv2DTranspose(filters=self.channels, kernel_size=5,
                               strides=2, padding='same',
                               kernel_initializer=RandomNormal(0, 0.02),
                               activation='tanh')(conv)
        
        model = Model(inputs=noise, outputs=conv)
        model.summary()

        return model
    
    def build_floyedhub_generator(self):
        '''
        Build Generator according to the following floyedhub exampe:
            https://github.com/ReDeiPirati/dcgan
        ! Pre-trained torch model is available for this implementation
        '''
        
        def gen_layer(input_, filters, kernel_size, strides, 
                  padding = 'same'):
            '''Layer used to build the generator'''
            out = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                  strides=strides, padding=padding,
                                  kernel_initializer=RandomNormal(0, 0.02),
                                  use_bias=False)(input_)
            out = Dropout(self.gen_dropout)(out)
            out = BatchNormalization(momentum=self.gen_momentum)(out)
            out = LeakyReLU(alpha=0)(out) # same as relu
            return out
        
        # Build the generator: 
        noise = Input(shape=self.latent_dim)  # is 1x1x100
        conv = gen_layer(noise, 512, 4, 1, 'valid') # becomes 4x4x512
        conv = gen_layer(conv, 256, 4, 2) # becomes 8x8x256
        conv = gen_layer(conv, 128, 4, 2) # becomes 16x16x128
        conv = gen_layer(conv, 64, 4, 2)  # becomes 32x32x64
        # becomes 64x64x3
        conv = Conv2DTranspose(filters=self.channels, kernel_size=4,
                              strides=2, padding='same',
                              kernel_initializer=RandomNormal(0, 0.02),
                              use_bias=False, activation='tanh')(conv)
        
        model = Model(inputs=noise, outputs=conv)
        model.summary()

        return model
    
    def build_discriminator(self):
        
        def dis_layer(filters, kernel_size, strides, input_):
            '''Layer used to build the discriminator'''
            out = Conv2D(filters=filters, kernel_size=kernel_size,
                         strides=strides, padding='same',
                         kernel_initializer=RandomNormal(0, 0.02),
                         use_bias=False)(input_)
            out = LeakyReLU(alpha=0.2)(out)
            out = Dropout(self.dis_dropout)(out)
            out = BatchNormalization(momentum=self.dis_momentum)(out)
            return out
        
        # Builds the discriminator: 
        img = Input(self.img_shape)       # is 64x64x3
        conv = dis_layer(64, 4, 2, img)   # becomes 32x32x64
        conv = dis_layer(128, 4, 2, conv) # becomes 16x16x128
        conv = dis_layer(256, 4, 2, conv) # becomes 8x8x256
        conv = dis_layer(512, 2, 2, conv) # becomes 4x4x512
        flat = Flatten()(conv)            # becomes flat
        sigm = Dense(1, activation='sigmoid')(flat) # gives output prob
        
        model = Model(inputs=img, outputs=sigm)
        model.summary()

        return model

    def train(self, data, epochs=100, batch_size=128, display_interval=100,
             model_save_interval = 500):
        k = 0
        n = data.shape[0]
        self.batcher = Batcher(data, batch_size, self.latent_dim, self.generator)
        
        while self.batcher.epoch <= epochs:
            # Gets batch
            real_batch, fake_batch, noise = self.batcher.next_batch()
            #  Train Generator
            g_loss = self.combined.train_on_batch(noise, self.batcher.gen_valid)
            #  Train Discriminator
            d_loss_real = self.discriminator.train_on_batch(real_batch, self.batcher.dis_valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_batch, self.batcher.dis_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Plot the progress
            print('[epoch: %d, imgs: %.2f%%][D loss: %f, acc_real: %.2f%%, acc_fake: %.2f%%] [G loss: %f]' % (self.batcher.epoch, (self.batcher.start/n)*100, d_loss[0], 100*d_loss_real[1], 100*d_loss_fake[1], g_loss))
                
            # If at save interval => save generated image samples
            if self.batcher.times_called % display_interval == 0:
                k += 1
                self.generate_imgs(k)
                
            if self.batcher.times_called % model_save_interval == 0:
                # Saves the generator and the combined model
                self.save()

    def generate_imgs(self, save = -1):
        '''
        Set save to any positive interger to also save the genered
        images with name res_'save'.
        '''
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r*c,) + self.latent_dim)
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = np.minimum(1, np.maximum(0,np.array(gen_imgs)))

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        plt.show()
        if save != -1:
            name = 'res_%d.png' % save
            path = os.path.join(self.paths['generated_imgs'], name)
            fig.savefig(path)
        plt.close()
    
    def save(self):
        # Saves the weights of the generator and discriminator on
        # the folder indicated by 'self.paths'
        self.generator.save_weights(self.paths['generator_weights'])
        self.discriminator.save_weights(self.paths['discriminator_weights'])
        
    def load(self):
        # Attempts to load the weights of the generator and discri-
        # minator from the folder indicated by 'self.paths'
        if os.path.exists(self.paths['generator_weights']):
            self.generator.load_weights(self.paths['generator_weights'])
        if os.path.exists(self.paths['discriminator_weights']):
            self.discriminator.load_weights(self.paths['discriminator_weights'])
    
    def delete(self):
        # Attempts to delete the weights of the generator and discri-
        # minator from the folder indicated by 'self.paths'
        if os.path.exists(self.paths['generator_weights']):
            os.remove(self.paths['generator_weights'])
        if os.path.exists(self.paths['discriminator_weights']):
            os.remove(self.paths['discriminator_weights'])
    

#################################
# Custom loss function and metric
#################################
        
import tensorflow as tf

def sigmoid_cross_entropy_with_logits(y_true, y_pred):
    # Gets logits as inputs for y_pred
    # See Tensorflow corresponding loss for more
    x = - tf.log(1. / y_pred - 1.)
        
    return tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y_true, logits=x)

def soft_binary_accuracy(y_true, y_pred):
    # Binary accuracy that takes into consideration
    # soft labels
    return tf.equal(tf.round(y_true), tf.round(y_pred))
