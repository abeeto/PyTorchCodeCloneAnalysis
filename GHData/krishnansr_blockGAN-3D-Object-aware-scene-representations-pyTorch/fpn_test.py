import os
import cv2
import torch
import pickle
import numpy as np
import os.path as osp
from torchvision import transforms
from fpn_od import SSODetector


BASE_PATH = r"/shared/home/c_sivarams/datasets/comp_cars"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "image"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "label"])

BASE_OUTPUT = "experiments_od"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
INFERENCE_PATH = os.path.sep.join([BASE_OUTPUT, "inference_result"])


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INIT_LR = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 32
LABELS = 1.0
BBOX = 1.0


def create_dirs(model_dir):
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

create_dirs(INFERENCE_PATH)

imagePaths = TEST_PATHS
imagePaths = open(imagePaths).read().strip().split("\n")

print("[INFO] loading object detector...")
model = torch.load(MODEL_PATH).to(DEVICE)
model.eval()
le = pickle.loads(open(LE_PATH, "rb").read())

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

for imagePath in imagePaths:
    print(f'inferring {imagePath}')
    image = cv2.imread(imagePath)

    overlayed = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))
    # convert image to PyTorch tensor, normalize it, flash it to the
    # current device, and add a batch dimension
    image = torch.from_numpy(image)
    image = transforms(image).to(DEVICE)
    image = image.unsqueeze(0)

    # label
    (boxPreds, labelPreds) = model(image)
    (startX, startY, endX, endY) = boxPreds[0]
    labelPreds = torch.nn.Softmax(dim=-1)(labelPreds)
    i = labelPreds.argmax(dim=-1).cpu()
    label = le.inverse_transform(i)[0]

    # orig = imutils.resize(orig, width=600)
    (h, w) = overlayed.shape[:2]
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(overlayed, f"class_{str(label)}", (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2)
    cv2.rectangle(overlayed, (startX, startY), (endX, endY),
                  (0, 255, 0), 2)

    res_path = os.path.sep.join([INFERENCE_PATH, os.path.basename(imagePath)])
    cv2.imwrite(res_path, overlayed)