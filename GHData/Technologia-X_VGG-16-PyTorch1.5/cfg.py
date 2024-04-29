'''
Configuration file.
'''
import os
from utils import get_classes, generate_img_label, create_dir, check_file_exist, get_device

####################################################### GENERAL CONFIGURATIONS #######################################################
CURR_FILE_PATH = '/'.join(os.path.realpath(__file__).split('/')[:-1]) #the absolute path of this file.
DEVICE = get_device() #identify the device to be used for training/evaluation
DATASET_PATH = CURR_FILE_PATH + '/dataset/' #absolute path of the dataset folder.
IMAGE_EXTS = ['jpg', 'png', 'jpeg', 'bmp'] #image file types.

MODEL_PATH = CURR_FILE_PATH +'/torch_model/' #trained model save path.
MODEL_NAME = 'vgg16-torch.pth'

create_dir(MODEL_PATH) #creates an empty directory to save the model after training if the folder does not exist.

#checks if the torch model exists.
TRAINED_MODEL_PRESENCE = check_file_exist(file_path=MODEL_PATH, file_name=MODEL_NAME)

####################################################### IMAGE CONFIGURATIONS #######################################################

RESIZED_IMAGE_SIZE = 224

#IMAGE AUGMENTATION CONFIGURATIONS

#probability of each augmentations. Set to 0 if you wish to disable the augmentation.
ROTATION_PROB = 0.5
SHEAR_PROB = 0.5
HFLIP_PROB = 0.5
VFLIP_PROB = 0
NOISE_PROB = 0.5

#providing one value will shear or rotate the image in the range of (-VALUE, +VALUE). A tuple or list with 2 elements can be provided as well to represent min and max value.
ROTATION_RANGE = 60 #equivalent to (-60, 60) or [-60, 60]
SHEAR_RANGE = 0.4 #equivalent to (-0.4, 0.4) or [-0.4, 0.4]
NOISE_MODE = ['gaussian', 'salt', 'pepper', 's&p', 'speckle'] #skimage noise modes.

####################################################### DATASET CONFIGURATIONS #######################################################
CLASSES, NUM_CLASSES = get_classes(dataset_folder_path=DATASET_PATH) #get all the classes names and the number of classes.

IMG_LABEL_LIST = generate_img_label(dataset_path=DATASET_PATH, classes=CLASSES, img_exts=IMAGE_EXTS)

TOTAL_DATA = len(IMG_LABEL_LIST)

NUM_WORKERS = 4 #number of workers to process the dataset loading.
DATA_SHUFFLE = True

####################################################### TRAINING CONFIGURATIONS #######################################################

BATCH_SIZE = 5
EPOCH = 1000
LEARNING_RATE = 1e-4
LR_DECAY_RATE = 0.99


