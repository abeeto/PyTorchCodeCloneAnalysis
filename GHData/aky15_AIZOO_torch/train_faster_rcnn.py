import torch
import torchvision
from faster_rcnn_utils.engine import train_one_epoch, voc_evaluate, coco_evaluate
from faster_rcnn_utils.AIZOODataset import AIZOODataset
from faster_rcnn_utils.transforms import get_transform
from faster_rcnn_utils import utils
import os

train_path = './train'
val_path = './val'
os.makedirs("checkpoints_faster_rcnn", exist_ok=True)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 3 classes, background, face，face_mask
num_classes = 3
BATCH_SIZE = 2
# use our dataset and defined transformations
dataset_train = AIZOODataset(train_path, get_transform(train=True))
dataset_val = AIZOODataset(val_path, get_transform(train=False))


# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=16,
    collate_fn=utils.collate_fn)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=16,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)  

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

# SGD
optimizer = torch.optim.SGD(params, lr=0.0003,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

# let's train it for   epochs
num_epochs = 31

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset    
    voc_evaluate(model, data_loader_val, device=device)    
    
    torch.save(model.state_dict(), f"checkpoints_faster_rcnn/faster_rcnn_ckpt_%d.pth" % epoch)
