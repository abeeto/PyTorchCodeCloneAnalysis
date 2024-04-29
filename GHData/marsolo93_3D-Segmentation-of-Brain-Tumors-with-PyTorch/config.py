import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = 3
BATCH_SIZE = 1
NUM_WORKERS = 4
SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8
EPOCHS = 200
WARMUP_STEPS = 100
LEARING_RATE_BEGIN = 1e-6
LEARNING_RATE_END = 1e-7

TRAIN_PATH = '/media/tensorist/Extreme SSD/brats2020/trainset.csv'
TEST_PATH = '/media/tensorist/Extreme SSD/brats2020/testset.csv'
CHECKPOINT = '/home/tensorist/project_files/tumor_segmentation/checkpoint'
LOAD_CHECKPOINT = '/home/tensorist/project_files/tumor_segmentation/checkpoint_224'



