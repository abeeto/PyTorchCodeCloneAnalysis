'''
Configuration file for model training.
'''
import os
from utils import get_classes, generate_img_label, create_dir, check_file_exist, get_device
from main_cfg import ARGS

####################################################### GENERAL CONFIGURATIONS #######################################################
CURR_FILE_PATH = '/'.join(os.path.realpath(__file__).split('/')[:-1]) #the absolute path of this file.
DEVICE = get_device() #identify the device to be used for training/evaluation
DATASET_PATH = CURR_FILE_PATH + ARGS.dataset_path #absolute path of the dataset folder.
IMAGE_EXTS = ARGS.image_exts #image file types.

MODEL_PATH = CURR_FILE_PATH + ARGS.model_path #trained model save path.
MODEL_NAME = ARGS.model_name

create_dir(MODEL_PATH) #creates an empty directory to save the model after training if the folder does not exist.

#checks if the torch model exists.
TRAINED_MODEL_PRESENCE = check_file_exist(file_path=MODEL_PATH, file_name=MODEL_NAME)

####################################################### IMAGE CONFIGURATIONS #######################################################

RESIZED_IMAGE_SIZE = ARGS.image_size

#IMAGE AUGMENTATION CONFIGURATIONS

#probability of each augmentations. Set to 0 if you wish to disable the augmentation.
ROTATION_PROB = ARGS.rotation_prob
SHEAR_PROB = ARGS.shear_prob
HFLIP_PROB = ARGS.hflip_prob
VFLIP_PROB = ARGS.vflip_prob
NOISE_PROB = ARGS.noise_prob

#providing one value will shear or rotate the image in the range of (-VALUE, +VALUE). A list with 2 elements can be provided as well to represent min and max value.
try:
    ROTATION_RANGE = list(ARGS.rotation_range)
except TypeError:
    ROTATION_RANGE = int(ARGS.rotation_range)

try:
    SHEAR_RANGE = list(ARGS.shear_range)
except TypeError:
    SHEAR_RANGE = float(ARGS.shear_range)

NOISE_MODE = ARGS.noise_mode #skimage noise modes.

####################################################### DATASET CONFIGURATIONS #######################################################
CLASSES, NUM_CLASSES = get_classes(dataset_folder_path=DATASET_PATH) #get all the classes names and the number of classes.

IMG_LABEL_LIST = generate_img_label(dataset_path=DATASET_PATH, classes=CLASSES, img_exts=IMAGE_EXTS)

TOTAL_DATA = len(IMG_LABEL_LIST)

NUM_WORKERS = ARGS.num_workers #number of workers to process the dataset loading.
DATA_SHUFFLE = ARGS.data_shuffle

####################################################### TRAINING CONFIGURATIONS #######################################################

BATCH_SIZE = ARGS.batch_size
EPOCH = ARGS.epoch
LEARNING_RATE = ARGS.learning_rate
LR_DECAY_RATE = ARGS.learning_rate_decay
PLOT_GRAPH = ARGS.plot_graph
