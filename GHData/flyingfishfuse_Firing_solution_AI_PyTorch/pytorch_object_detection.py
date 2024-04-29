#!"C:\Python36\python.exe"

# thinking about abandoning the tensorflow shit in favor of pytorch... seems better for some reason :p
import os
import sys
import glob
import torch
import numpy as np
from models import *
from PIL import Image
import tensorflow as tf
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from utils import *import os, sys, time, datetime, random
from torch.autograd import Variableimport matplotlib.pyplot as plt

'''
First of all, you need to get all your training images together, using this folder structure (folder names are in italics):

Main Folder
--- data
    --- dataset name
        --- images
            --- img1.jpg
            --- img2.jpg
            ..........
        --- labels
            --- img1.txt
            --- img2.txt
            ..........
        --- train.txt
        --- val.txt
'''


GPU_MEMORY_LIMIT_PER_GPU           = 1024

MODEL_SELECT             = "yolo"
MODEL_NAME               = 'ssd_inception_v2_coco_2018_1_28'
PATH_TO_MODEL            = 'object_detection/models' + MODEL_NAME
PATH_TO_LABELS           = 'object_detection/data/mscoco_label_map.pbtxt'
LABEL_MAP                = label_map_util.load_labelmap(PATH_TO_LABELS)
CATEGORIES               = label_map_util.convert_label_map_to_categories(LABEL_MAP, max_num_classes=NUM_CLASSES, use_display_name=True)
CATEGORY_INDEX           = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
DETECTION_MODEL          = tf.keras.models.load_model(model_name)
PATH_TO_TEST_IMAGES_DIR  = 'test_images'
TEST_IMAGE_PATHS         = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 8) ]

path_to_classes          = './data/coco.names'
path_to_weights          = './checkpoints/yolov3.tf'
image_resize_size        = 416
INPUT_image              = './data/girl.png'
output                   = '.output.jpg'
num_classes              = 80
yolo_iou_threshold       = 0.5
yolo_score_threshold     = 0.5
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

# setting the device to use
GPU_DEVICE = torch.device("cuda:0")

class DistributedModel(nn.Module):
''' Implements a distributed multi-processor approach utilizing boththe GPU and CPU''' 
    def __init__(self):
        super().__init__(
            embedding=nn.Embedding(1000, 10),
            rnn=nn.Linear(10, 10).to(device),
        )

    def forward(self, x):
        # Compute embedding on CPU
        x = self.embedding(x)

        # Transfer to GPU
        x = x.to(device)

        # Compute RNN on GPU
        x = self.rnn(x)
        return x
if MODEL_SELECT == "yolo":
    MODEL_NAME  = 'yolov3.weights'
else :
    MODEL_NAME  = "ssd_inception_v2_coco_2018_1_28.weights" 
def train_model()
''' Trains the model'''
    current_dir = "./data/artifacts/images"
    split_pct = 10  # 10% validation set
    file_train = open("data/artifacts/train.txt", "w")  
    file_val = open("data/artifacts/val.txt", "w")  
    counter = 1  
    index_test = round(100 / split_pct)  
    for fullpath in glob.iglob(os.path.join(current_dir, "*.JPG")):  
        title, ext = os.path.splitext(os.path.basename(fullpath))
        if counter == index_test:
            counter = 1
            file_val.write(current_dir + "/" + title + '.JPG' + "\n")
        else:
            file_train.write(current_dir + "/" + title + '.JPG' + "\n")
            counter = counter + 1
    file_train.close()
    file_val.close()

def detect_image(img):
'''Detector for the yolo set. Feed it an image, machine gun style!'''
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0), 
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, 
                        conf_thres, nms_thres)
    return detections[0]

# load image and get detections
img_path = "images/blueangels.jpg"
prev_time = time.time()
img = Image.open(img_path)
detections = detect_image(img)
inference_time = datetime.timedelta(seconds=time.time() - prev_time)
print ('Inference Time: %s' % (inference_time))# Get bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]img = np.array(img)
plt.figure()
fig, ax = plt.subplots(1, figsize=(12,9))
ax.imshow(img)pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
unpad_h = img_size - pad_y
unpad_w = img_size - pad_xif detections is not None:
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)
    # browse detections and draw bounding boxes
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
        color = bbox_colors[int(np.where(
             unique_labels == int(cls_pred))[0])]
        bbox = patches.Rectangle((x1, y1), box_w, box_h,
             linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        plt.text(x1, y1, s=classes[int(cls_pred)], 
                color='white', verticalalignment='top',
                bbox={'color': color, 'pad': 0})
plt.axis('off')
# save image
plt.savefig(img_path.replace(".jpg", "-det.jpg"),        
                  bbox_inches='tight', pad_inches=0.0)
plt.show()