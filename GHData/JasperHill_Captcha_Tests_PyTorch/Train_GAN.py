from __future__ import absolute_import, division
import os
import sys

sys.path.append(os.path.join(os.getcwd(),'build_tools'))

import re
import time
import pathlib
import string
import math
import MODEL_CONSTANTS
import torch
import torch.nn                as nn
import torch.nn.functional     as F
import torch.optim             as optim
import numpy                   as np
import matplotlib.pyplot       as plt
import torchvision.transforms  as transforms

from Captcha_GAN               import Generator, Discriminator
from optparse                  import OptionParser
from torch.autograd            import Function
from skimage                   import io
from torch.utils.data          import Dataset, DataLoader



## input and output specs
## alphanumeric contains all allowed captcha characters
## D is the full length of each vectorized captcha string
## N is the number of characters in each captcha string

alphanumeric = string.digits + string.ascii_lowercase
D            = len(alphanumeric)
N            = 5

IMG_HEIGHT   =   MODEL_CONSTANTS.IMG_HEIGHT
IMG_WIDTH    =    MODEL_CONSTANTS.IMG_WIDTH
IMG_CHANNELS = MODEL_CONSTANTS.IMG_CHANNELS

#############################################################################################################################
##  preprocess data and train the GAN
##  
#############################################################################################################################

"""
This project is an adversarial generative neural network inspired by the work of
Ye et al., Yet Another Captcha Solver: An Adversarial Generative Neural Network Based Approach

The configuration herein is much simpler and less robust, but it follows the same method:
A relatively small sample of captcha images are presented to a network containing a generator,
attempting to reproduce such captcha images from their corresponding labels and a discriminator,
attempting to discern authentic and synthetic images. The two work toward opposing goals, and
training ceases when the discriminator is unable to correctly classify a certain fraction of the
inputs.

Unlike conventional GANs, this model features custom-built layers that function like
quantum-mechanical operators. The generator and discriminator are nearly inversely configured w.r.t
each other. The discriminator does feature a sigmoid activation on its output to constrain the
output.

TODO: build a solver
"""

## note: one of the png files from this source is improperly named; mv 3bnfnd.png 3bfnd.png
sampledir = os.path.join('.','samples')
p         = pathlib.Path(sampledir)

jpg_count = len(list(p.glob('*/*.jpg')))
png_count = len(list(p.glob('*/*.png')))
NUM_OF_IMAGES = jpg_count + png_count

train_dir = os.path.join(sampledir, 'training_samples')
test_dir = os.path.join(sampledir, 'testing_samples')

print('jpg_count: {}, png_count: {}'.format(jpg_count, png_count))


#############################################################
## processing and mapping functions for the files
#############################################################

## label generation and vectorization ##

alphanumeric = MODEL_CONSTANTS.alphanumeric
D = len(alphanumeric)
N = MODEL_CONSTANTS.N

def char_to_vec(char):
    vec = np.zeros(D, dtype=np.double)
    match = re.search(char,alphanumeric)
    vec[match.span()[0]] = 1
    return vec

def char_to_int(char):
    return re.search(char, alphanumeric).span()[0]
    
def string_to_mat_and_sparse_mat(string):
    N = len(string)
    mat = np.zeros([N,D], dtype=np.double)
    sparse_mat = np.zeros([N,D,D], dtype=np.double)

    d = 0
    for char in string:
        mat[d] = char_to_vec(char)
        sparse_mat[d] = np.tensordot(mat[d],mat[d],axes=0)
        d += 1

    return mat,sparse_mat

def string_to_dense_mat(string):
    vec = np.empty(N)
    dense_mat = np.empty([1,N,N])
    for i in range(N):
        vec[i] = char_to_int(string[i])

    dense_mat[0] = np.tensordot(vec,vec,axes=0)/(math.pow(D,2))

    return dense_mat
        
def NN_mat_to_string(nnmat):
    string = ''

    for i in range(N):
        idx = tf.argmax(nnmat[i])
        string += alphanumeric[idx]

    return string

## transform matrices from the dataset back to strings for visualization
def mat_to_string(mat):
     string = ''
     npmat = mat.numpy()

     for i in range(N):
         for j in range(D):
             if (npmat[i][j] == 1):
                 string += alphanumeric[j]
                 break

     return string

def generate_labels(filename):
    parts = re.split('\.',filename)
    string_label = parts[0]

    mat_label, sparse_label = string_to_mat_and_sparse_mat(string_label)
    dense_mat = string_to_dense_mat(string_label)

    return string_label, mat_label, sparse_label, dense_mat


## create a dataset subclass
class CaptchaDataset(Dataset):
    def __init__(self, imgdir, transform=None):
        super(Dataset, self).__init__()
        self.imgdir = imgdir
        self.fnames = os.listdir(imgdir)
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        fname                                            = self.fnames[idx]
        imgpath                                          = os.path.join(self.imgdir, fname)
        img                                              = io.imread(imgpath)
        string_label, mat_label, sparse_label, dense_mat = generate_labels(fname)

        if self.transform: img = self.transform(img)
        
        ## pngs contain alpha channels while jpgs do not ## so return img[:3] to clip the alpha channel if present
        img = img[:3]


        return {'images': img, 'string labels': string_label, 'mat labels': mat_label, 'sparse labels': sparse_label, 'dense mats': dense_mat}

## auxiliary function to visualize the data
def show_batch(imgs, str_labels, guesses, filename):
    plt.figure(num = 0, figsize=(10,10))
    batch_size = len(imgs)
    if (batch_size > 10): batch_size = 10
    
    for n in range(batch_size):
        str_label = str_labels[n]
        img = imgs[n]
        img = torch.transpose(img,0,1)
        img = torch.transpose(img,1,2)
        if guesses is not None: guess = NN_mat_to_string(guesses[n])

        ax = plt.subplot(np.ceil(batch_size/2),2,n+1)
        plt.imshow(img)

        if guesses is not None: plt.title('guess: {}'.format(guess))
        else:                   plt.title(str(str_label))

        plt.axis('off')
        plt.savefig(filename+'.pdf')
        plt.close(0)


## auxiliary function to compute discriminator accuracy
def accuracy(guesses, answers, epsilon):
    count = 0
    total = len(guesses)
    
    for i in range(total):
        guess = guesses[i].detach().numpy()
        answer = answers[i].detach().numpy()

        if (math.fabs(guess - answer) <= epsilon): count += 1

    return (count/total * 100)
#############################################################
##  prepare dataset
#############################################################

EPOCHS      =  range(MODEL_CONSTANTS.NUM_EPOCHS)
BATCH_SIZE  =        MODEL_CONSTANTS.BATCH_SIZE
NUM_WORKERS =                                 1
EPSILON     =           MODEL_CONSTANTS.EPSILON

train_ds = CaptchaDataset(train_dir, transform=transforms.Compose([transforms.ToTensor()]))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

test_ds = CaptchaDataset(test_dir, transform=transforms.Compose([transforms.ToTensor()]))
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


GEN_SAVE_PATH  =  MODEL_CONSTANTS.GEN_SAVE_PATH
DISC_SAVE_PATH = MODEL_CONSTANTS.DISC_SAVE_PATH

generator = Generator(input_shape=[N,D,D])
discriminator = Discriminator(input_shape=[IMG_CHANNELS,IMG_HEIGHT,IMG_WIDTH])

generator.load_state_dict(torch.load(GEN_SAVE_PATH))
discriminator.load_state_dict(torch.load(DISC_SAVE_PATH))

## an initial learning rate of 1e-4 seems ideal for the custom layers
GEN_LR          =                           MODEL_CONSTANTS.GEN_LR
DISC_LR         =                          MODEL_CONSTANTS.DISC_LR
gen_optimizer   =     optim.Adam(generator.parameters(), lr=GEN_LR)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=DISC_LR)

gen_train_loss_hist  = []
disc_train_loss_hist = []
disc_train_acc_hist = []

gen_test_loss_hist   = []
disc_test_loss_hist  = []
disc_test_acc_hist = []

train_imax = len(train_dl)
test_imax = len(test_dl)

status_labels = ['relative gen training loss',
                'relative gen testing loss',
                 'relative disc training loss',
                 'disc testing accuracy']


## train the GAN

print('|{: ^10}'.format('epoch'), end='')
for lbl in status_labels:
    print('|{: ^30}'.format(lbl), end='')
print('|')

for epoch in EPOCHS:
    gen_loss = 0.0
    disc_loss = 0.0
    disc_acc = 0.0
    i = 0

    for data in train_dl:
    
        auth_imgs       = data['images']
        string_labels   = data['string labels']
        mat_labels      = data['mat labels']
        sparse_labels   = data['sparse labels']
        dense_mats      = data['dense mats']

        synth_imgs = generator(sparse_labels).to(torch.double)
        
        if (i >= train_imax): break
        gen_optimizer.zero_grad()
        disc_optimizer.zero_grad()

        synth_imgs = generator(sparse_labels).to(torch.double)
        synth_guesses = discriminator(synth_imgs).to(torch.double)
        auth_guesses = discriminator(auth_imgs).to(torch.double)

        ref_auth_guesses = 1 - (EPSILON * torch.rand_like(auth_guesses).to(torch.double))
        ref_synth_guesses = EPSILON * torch.rand_like(synth_guesses).to(torch.double)

        ## r ensures mse and binary crossentropy are weighted equally
        r = torch.div(F.mse_loss(synth_imgs, auth_imgs.to(torch.double), reduction='sum'),
                         F.binary_cross_entropy(synth_guesses, ref_auth_guesses, reduction='sum'))
        
        generator_loss = torch.add(F.mse_loss(synth_imgs, auth_imgs.to(torch.double), reduction='sum'),
                                   r*F.binary_cross_entropy(synth_guesses, ref_auth_guesses, reduction='sum'))
        
        discriminator_loss  = torch.add(F.binary_cross_entropy(synth_guesses, ref_synth_guesses, reduction='sum'),
                                        F.binary_cross_entropy(auth_guesses, ref_auth_guesses, reduction='sum'))

        discriminator_acc = 0.5*(accuracy(synth_guesses, ref_synth_guesses, EPSILON) + accuracy(auth_guesses, ref_auth_guesses, EPSILON))

        generator_loss.backward(retain_graph=True)
        discriminator_loss.backward(retain_graph=True)
        
        gen_optimizer.step()
        disc_optimizer.step()

        gen_loss += generator_loss.item()
        disc_loss += discriminator_loss.item()
        disc_acc += discriminator_acc
        i += 1

    if (epoch == 0):
        gen_train_loss_0 = gen_loss
        disc_train_loss_0 = disc_loss        
        
    gen_train_loss_hist.append(gen_loss/gen_train_loss_0)
    disc_train_loss_hist.append(disc_loss/disc_train_loss_0)
    disc_train_acc_hist.append(disc_acc/(train_imax+1))

    gen_loss = 0.0
    disc_loss = 0.0
    disc_acc = 0.0
    
    ## test the GAN at the end of each training epoch
    with torch.no_grad():
        for i, data in enumerate(test_dl, 0):
            if (i >= test_imax): break
            auth_imgs       = data['images']
            string_labels   = data['string labels']
            mat_labels      = data['mat labels']
            sparse_labels   = data['sparse labels']
            dense_mats      = data['dense mats']
            
            synth_imgs = generator(sparse_labels).to(torch.double)
            synth_guesses = discriminator(synth_imgs).to(torch.double)
            auth_guesses = discriminator(auth_imgs).to(torch.double)
            
            ref_auth_guesses = 1 - (EPSILON * torch.rand_like(auth_guesses).to(torch.double))
            ref_synth_guesses = EPSILON * torch.rand_like(synth_guesses).to(torch.double)

            r = torch.div(F.mse_loss(synth_imgs, auth_imgs, reduction='sum'),
                          F.binary_cross_entropy(synth_guesses, ref_auth_guesses, reduction='sum'))                    
            
            generator_loss = torch.add(F.mse_loss(synth_imgs, auth_imgs, reduction='sum'),
                                       r*F.binary_cross_entropy(synth_guesses, ref_auth_guesses, reduction='sum'))
            
            discriminator_loss  = torch.add(F.binary_cross_entropy(synth_guesses, ref_synth_guesses, reduction='sum'),
                                            F.binary_cross_entropy(auth_guesses, ref_auth_guesses, reduction='sum'))

            discriminator_acc = 0.5*(accuracy(synth_guesses, ref_synth_guesses, EPSILON) + accuracy(auth_guesses, ref_auth_guesses, EPSILON))
            
            gen_loss += generator_loss.item()
            disc_loss += discriminator_loss.item()
            disc_acc += discriminator_acc

        if (epoch == 0):
            gen_test_loss_0 = gen_loss
            disc_test_loss_0 = disc_loss

        gen_test_loss_hist.append(gen_loss/gen_test_loss_0)
        disc_test_loss_hist.append(disc_loss/disc_test_loss_0)
        disc_test_acc_hist.append(disc_acc/(test_imax+1))

        if (epoch == EPOCHS[-1]):
            show_batch(synth_imgs, string_labels, guesses=None, filename='synthetic_captchas')
        
    ## print epoch results
    nums = [gen_train_loss_hist[-1],
            gen_test_loss_hist[-1],
            disc_train_loss_hist[-1],
            disc_test_acc_hist[-1]]
    print('|{: >10d}'.format(epoch), end='')
    for num in nums:
        print('|{: >30.3f}'.format(num), end='')
    print('|')


torch.save(generator.state_dict(), GEN_SAVE_PATH)
torch.save(discriminator.state_dict(), DISC_SAVE_PATH)        

    
plt.figure(num=1, figsize=(10,10))
plt.plot(EPOCHS, np.log10(np.asarray(gen_train_loss_hist)), 'bo', label='training loss')
plt.plot(EPOCHS, np.log10(np.asarray(gen_test_loss_hist)), 'go', label='testing loss')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig('gen_losses.pdf')
plt.close(1)

