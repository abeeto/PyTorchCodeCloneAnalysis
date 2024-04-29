'''
Configuration file for model evaluation.
'''
import os
from utils import get_device, check_file_exist
from main_cfg import ARGS

CURR_FILE_PATH = '/'.join(os.path.realpath(__file__).split('/')[:-1]) #the absolute path of this file.
DEVICE = get_device() #identify the device to be used for training/evaluation
FEAT_MODEL_PATH = CURR_FILE_PATH + ARGS.model_path #trained model save path.
FEAT_MODEL_NAME = ARGS.model_name
CLASSIFIER_MODEL_PATH = FEAT_MODEL_PATH
CLASSIFIER_MODEL_NAME = ARGS.save_model_name

#checks if the torch model exists.
TRAINED_FEAT_MODEL_PRESENCE = check_file_exist(file_path=FEAT_MODEL_PATH, file_name=FEAT_MODEL_NAME)
TRAINED_CLASSIFIER_MODEL_PRESENCE = check_file_exist(file_path=CLASSIFIER_MODEL_PATH, file_name=CLASSIFIER_MODEL_NAME)


RESIZED_IMAGE_SIZE = ARGS.image_size

CLASS_FILE = ARGS.class_file
CLASSES = []

#Get the classes name from the .txt file.
OPEN_CFILE = open(CLASS_FILE, 'r')

#Reads every line in the file and append the name into the list.
for line in OPEN_CFILE:
    CLASSES.append(line.rstrip()) #strip the newline.

CLASSES.sort() #sort ascending order. IMPORTANT!
NUM_CLASSES = len(CLASSES)
FEATURE_INPUT_SIZE = 512 * (RESIZED_IMAGE_SIZE//(2**5))**2 #final conv layer output channel x subsampled image height x subsampled image width
