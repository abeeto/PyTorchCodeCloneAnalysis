import torch
import os 


# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files

BASE_PATH = r"C:\Users\richa\OneDrive\Documentos\Object-detector-from-scratch-in-PyTorch"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])

BASE_OUTPUT = r"C:\Users\richa\OneDrive\Documentos\Object-detector-from-scratch-in-PyTorch\output"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATH = os.path.sep.join([BASE_OUTPUT, "test_path.txt"])


# Testing GPU is available   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True if DEVICE == "cuda" else False

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# initialize our initial learning rate, number of epochs to train
# for, and the batch size

INIT_LR = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 32
    
# specify the loss weights
LABELS = 1.0
BBOX = 1.0




 