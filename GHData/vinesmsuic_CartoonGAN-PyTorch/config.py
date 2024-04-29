import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = os.path.join("data", "train")
TRAIN_CARTOON_DIR = os.path.join(TRAIN_DIR, "cartoon")
TRAIN_CARTOON_EDGE_DIR = os.path.join(TRAIN_DIR, "cartoon_edge_pair")
TRAIN_PHOTO_DIR = os.path.join(TRAIN_DIR, "photo")

VAL_DIR = os.path.join("data", "val")
VAL_PHOTO_DIR = os.path.join(VAL_DIR, "photo")

RESULT_TRAIN_DIR = os.path.join("results", "train")
RESULT_TEST_DIR = os.path.join("results", "test")

BATCH_SIZE = 1
LEARNING_RATE = 2e-4

LAMBDA_ADV = 1
LAMBDA_CONTENT = 10

NUM_WORKERS = 4
IMAGE_SIZE = 256

PRETRAIN_EPOCHS = 10
NUM_EPOCHS = 100

LOAD_MODEL = True
SAVE_MODEL = True
SAVE_MODEL_FREQ = 5

CHECKPOINT_FOLDER = "checkpoints"
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
VGG_WEIGHTS = "vgg19-dcbb9e9d.pth"

LOAD_CHECKPOINT_DISC = "i_disc.pth.tar"
LOAD_CHECKPOINT_GEN = "i_gen.pth.tar"

SAVE_IMG_PER_STEP = 200

transform_cartoon_pairs = A.Compose(
    #additional_targets: apply same augmentation on both images
    [   
        #A.RandomCrop(width=IMAGE_SIZE*1.2, height=IMAGE_SIZE*1.2),
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        #A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ], 
    additional_targets={"image0": "image"},
)

transform_input = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        #A.HorizontalFlip(p=0.5),
        #A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)

transform_test = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)