import torch
import torchvision
from faster_rcnn_utils.engine import evaluate
from faster_rcnn_utils.AIZOODataset import AIZOODataset
from faster_rcnn_utils.transforms import get_transform
from faster_rcnn_utils import utils
import os

test_path = './test'

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 3 classes, background, face，face_mask
num_classes = 3
BATCH_SIZE = 2
# use our dataset and defined transformations
dataset_test = AIZOODataset(test_path, get_transform(train=False))

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=16,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
model.load_state_dict(torch.load('checkpoints_faster_rcnn/faster_rcnn_ckpt_30.pth'))
# move model to the right device
model.to(device)

# evaluate on the test dataset    
evaluate(model, data_loader_test, device=device)    
    
