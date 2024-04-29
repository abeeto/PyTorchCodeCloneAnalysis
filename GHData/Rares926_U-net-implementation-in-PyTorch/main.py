
#----------------------------#
#     Necessary imports      #
#----------------------------#

import torch
import albumentations as A  #python library for image augmentations
from albumentations.pytorch import ToTensorV2 #Convert image and mask to torch.Tensor and divide by 255 if image or mask are uint8 type
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from tqdm import tqdm #for the progress bar 

#----------------------------#
# Internal framework imports #
#----------------------------#


from utils.data_load import get_loaders
from utils.utils import save_checkpoint,save_predictions,load_checkpoint
from utils.dice import check_dice_score
from core.model import UNet
# Typing imports imports
from torch import Tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR  = "Z:\Xperi\semantic_segmentation\data/train_images"
TRAIN_MASK_DIR = "Z:\Xperi\semantic_segmentation\data/train_masks"
VAL_IMG_DIR    = "Z:\Xperi\semantic_segmentation\data/val_images"
VAL_MASK_DIR   = "Z:\Xperi\semantic_segmentation\data/val_masks"

IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH  = 240  # 1918 originally

LOAD_MODEL=True

EPOCHS       = 3
BATCH_SIZE   = 1    #32
NUM_WORKERS  = 2    #how many subprocesses to use for data loading
PIN_MEMORY   = False #If True, the data loader will copy Tensors into CUDA pinned memory before returning them

def train(loader:Tensor,model,optimizer,loss_fn):

    loop=tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data    = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        optimizer.zero_grad()

        # FORWARD
        predictions = model(data)
        # predictions=torch.nn.functional.softmax(predictions)
        loss = loss_fn(predictions, targets)  

        # BACKWARD
        loss.backward()
        optimizer.step()
        
        # UPDATE TQDM LOOP
        loop.set_postfix(loss=loss.item())

def run():


    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.1),
            # A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )




    #----------------------------#
    #       LOSS FUNCTION        #
    #----------------------------#
    loss_fn = CrossEntropyLoss()

    #----------------------------#
    #           MODEL            #
    #----------------------------#
    model=UNet(n_channels=3, n_classes=2).to(DEVICE)

    #----------------------------#
    #        OPTIMIZER           #
    #----------------------------#
    optimizer = optim.SGD(model.parameters(), lr=4e-4,momentum=0.9)

    #----------------------------#
    #     CREATE THE LOADERS     #
    #----------------------------#
    train_loader,val_loader = get_loaders(TRAIN_IMG_DIR,TRAIN_MASK_DIR,VAL_IMG_DIR,VAL_MASK_DIR,BATCH_SIZE,train_transform,val_transforms,NUM_WORKERS,PIN_MEMORY)

    #----------------------------#
    #       START TRAINING       #
    #----------------------------#

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        check_dice_score(val_loader,model,DEVICE)
        save_predictions(val_loader,model,folder="Z:\Xperi\semantic_segmentation\U-net-implementation-in-PyTorch\saved predictions",device=DEVICE,batch_size=BATCH_SIZE)



    for epoch in range(EPOCHS):

        train(train_loader, model, optimizer, loss_fn)

        #----------------------------#
        #   SAVE THE TRAINED MODEL   #
        #----------------------------#
        checkpoint = {"epoch":epoch,
                      "model_state_dict" : model.state_dict(),
                      "optimizer_state_dict"  : optimizer.state_dict()}
        save_checkpoint(checkpoint)

        #----------------------------#
        #       CHECK ACCURACY       #
        #----------------------------#
        check_dice_score(val_loader, model, device=DEVICE)

        #----------------------------#
        #   SAVE SOME INFERENCES     #
        #----------------------------#
        save_predictions(val_loader,model,folder="Z:\Xperi\semantic_segmentation\U-net-implementation-in-PyTorch\saved predictions",device=DEVICE,batch_size=BATCH_SIZE)




if __name__ == "__main__":
    run()