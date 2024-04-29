from torch import cuda
from pathlib import Path

device = 'cuda:0' if cuda.is_available() else 'cpu'
IMAGE_PATH = 'DIV2K_train_HR'
IMAGE_SAVE_PATH = Path('images')
UPSCALE_FACTOR = 4
HR_SIZE = 200
LR_SIZE = HR_SIZE//UPSCALE_FACTOR
NUM_EPOCHS = 500
BATCH_SIZE = 16
NUM_BATCHES = 800/BATCH_SIZE
GEN_PATH = 'model/generator.pth.tar'
DISC_PATH = 'model/discriminator.pth.tar'