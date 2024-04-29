# -*- coding: utf-8 -*-
"""

@author: Haidar Almubarak
"""

import pandas as pd
from utilities import get_generator
import time
import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument("--save_weights_path", type = str,default="./" )

parser.add_argument("--train_images", type = str, default = "./RawData/images/")


parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )

parser.add_argument('--validate',action='store_true')
parser.add_argument("--val_images", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 100 )
parser.add_argument("--continue_from", type = int, default = 0 )

parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--val_batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "zf_unet_224")
parser.add_argument("--optimizer_name", type = str , default = "Adadelta")
parser.add_argument("--learningrate", type = float , default = 0.0001)

args = parser.parse_args()

# read the arguments either from a file using @filename as argument or using
# arguments directly 
train_images_path = args.train_images
train_batch_size = args.batch_size
input_height = args.input_height
input_width = args.input_width

validate = args.validate

save_weights_path = args.save_weights_path
epochs = args.epochs
continue_from = args.continue_from

load_weights = args.load_weights

optimizer_name = args.optimizer_name
learningrate = args.learningrate
model_name = args.model_name

totalEpochs = continue_from + epochs

if validate:
    val_images_path = args.val_images
    val_batch_size = args.val_batch_size
#
model = .....
# if load weight is provided load the weight to the model
if load_weights != '':
    .... # optional

# set the optimizer
if(optimizer_name == 'Adam'):
    optim = optimizers.Adam(lr=learningrate)
elif(optimizer_name=='Adadelta'):
    optim = optimizers.Adadelta(lr=learningrate)
elif(optimizer_name=='Adagrad'):
    optim = optimizers.Adagrad(lr=learningrate)
elif(optimizer_name=='RMSprop'):
    optim = optimizers.RMSprop(lr=learningrate)
elif(optimizer_name=='SGD'):
    optim = optimizers.SGD(lr=learningrate)
elif(optimizer_name=='Adamax'):
    optim = optimizers.Adamax(lr=learningrate)
else:
    optim = optimizers.Adadelta(lr=learningrate)

#%%
# get training and validation image generator form the input directory

trainGen, train_steps = get_generator(train_images_path,
                              train_segs_path,
                              batch_size = train_batch_size,
                              fill_mode="reflect")
if validate:
    validateGen,validat_steps = get_generator(val_images_path,
                                  val_segs_path,
                                  batch_size = val_batch_size,
                                  width_shift_range=0,
                                  height_shift_range=0,
                                  horizontal_flip=False,
                                  rotation_range=0,
                                  zoom_range=0)
#%%
start_time = time.time()

# start the training
totalEpochs = continue_from + epochs

if validate:
    # save if validation is correct
else:
    # just train

# save the model
torch.save(save_weights_path+ model_name + str( totalEpochs )+'.h5')


print("Elapsed time {}".format(time.time()-start_time))