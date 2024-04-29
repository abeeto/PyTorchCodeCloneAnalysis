import torch
from torch.utils.tensorboard import SummaryWriter


# 1 - segmentation | 2 - detection | 3 - instance segmentation | 4 - keypoints detection
TASK = 2

#----------------------GENERAL TRAINING CONFIGURATIONS-------------------------#
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter()
NUM_EPOCHS = 20
BATCH_SIZE = 2
LR = 0.005
IMAGE_HEIGH = 512
IMAGE_WIDTH = 512
TRAIN_RATIO = 0.85
CHECKPOINTS_PATH = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/checkpoints/'
N_CLASSES = 3
THRESHOLD = 0.6

#--------PARTICULAR TRAININT CONFIGURATIONS FOR EACH INDIVIDUAL TASK------------#
#-------------segmentation-----------#
if TASK == 1:
    IMAGES_PATH = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_segmentation/images/'
    ANNOTATIONS_PATH = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_segmentation/annotations/'
    LOG_DIR = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_segmentation/logs/'
    CHECKPOINT_FOR_INFERENCE = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/checkpoints/segmentation/checkpoint_18.pth'
#--------------detection-------------
if TASK == 2:
    IMAGES_PATH = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_detection/images/'
    ANNOTATIONS_PATH = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_detection/annotations/'
    LOG_DIR = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_detection/logs/'
    CHECKPOINT_FOR_INFERENCE = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/checkpoints/detection/checkpoint_detector_2.pth'
    LABELS_MAP= { 'banana':1, 'orange':2, 'apple': 3}

if TASK == 3:
    IMAGES_PATH = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_instance_segmentation/PNGImages/'
    ANNOTATIONS_PATH = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_instance_segmentation/Annotation/'
    MASKS_PATH = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_instance_segmentation/PedMasks/'
    LOG_DIR = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_detection/logs/'
    CHECKPOINT_FOR_INFERENCE = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/checkpoints/instance_segmentation/checkpoint_instance_segmentation.pth'
    LABELS_MAP = {'Human':1}