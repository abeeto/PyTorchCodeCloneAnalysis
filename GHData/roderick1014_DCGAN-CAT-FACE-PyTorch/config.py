import torch
from PIL import Image

ROOT_DIR = "Dataset/"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using "+ str(device) + " for training. :)")
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"

SAVE_MODEL = False
LOAD_MODEL = False

G_LEARNING_RATE = 1e-4 #1e-4
D_LEARNING_RATE = 1e-4

BATCH_SIZE = 256
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 200000
FEATURES_CRITIC = 64  #64
FEATURES_GEN = 64  #64
CRITIC_ITERATIONS = 1