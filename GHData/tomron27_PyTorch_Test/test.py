import os
import torch
from vgg import *
from PIL import Image
from torchvision import transforms
from data_loaders import *
from data_utils import validate
from os.path import join
from sklearn.metrics import multilabel_confusion_matrix as MCM
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)
handler = logging.FileHandler('test.py.log')
handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%Y-%m-%d %H:%M:%S'))
logger.setLevel(logging.INFO)
logger.addHandler(handler)

logger.info("--- test.py Log begin ---")

# Paths
base_dir = "/home/tomron27@st.technion.ac.il/"
project_dir = join(base_dir, "projects/PyTorch_Test/")
data_base_dir = join(base_dir, "projects/ChestXRay/data/fetch/")
images_path = join(data_base_dir, "images/")

val_metadata_path = join(data_base_dir, "validation_metadata.csv")
train_metadata_path = join(data_base_dir, "train_metadata.csv")
test_metadata_path = join(data_base_dir, "test_metadata.csv")

model_dir = join(project_dir, "models/")

# Params
batch_size = 1
input_size = 1024
resize_factor = 2
resize = input_size//resize_factor
tau = 0.3
print_interval = 2000

# Transformations
trans_list = []
if resize_factor > 1:
    trans_list += [transforms.Resize(resize)]

trans_list += [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
trans = transforms.Compose(trans_list)

# Dataloaders
data = ChestXRayDataset(csv_file=test_metadata_path,
                             root_dir=images_path,
                             transform=trans)

data_loader = torch.utils.data.DataLoader(dataset=data,
                                          batch_size=batch_size,
                                          shuffle=False)

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = vgg16_bn(size=512, num_classes=8)
model.to(device)

checkpoint = torch.load(join(model_dir, "20_epochs", "vgg_16_bn_norm_epoch_5.pt"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Validate
validate(model, device, data_loader, data.labels_dict, logger)
