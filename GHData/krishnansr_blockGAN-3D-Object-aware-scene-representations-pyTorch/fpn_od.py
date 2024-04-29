import os
import cv2
import glob
import torch
import pickle
import time
import numpy as np
import os.path as osp

from tqdm import tqdm
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # specify GPUs to use

BASE_PATH = r"/local/scratch/c_sivarams/datasets/comp_cars"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "image"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "label"])

BASE_OUTPUT = "experiments_od"
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INIT_LR = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 32
LABELS = 1.0
BBOX = 1.0


def create_dirs(model_dir):
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

create_dirs(BASE_OUTPUT)
create_dirs(PLOTS_PATH)


class CarsDataset(Dataset):
    # initialize the constructor
    def __init__(self, tensors, transforms=None):
        self.tensors = tensors
        self.transforms = transforms

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        image = self.tensors[0][index]
        label = self.tensors[1][index]
        bbox = self.tensors[2][index]
        image = image.permute(2, 0, 1)
        if self.transforms:
            image = self.transforms(image)
        return (image, label, bbox)


class SSODetector(Module):
    def __init__(self, baseModel, numClasses):
        super(SSODetector, self).__init__()
        # initialize the base model and the number of classes
        self.baseModel = baseModel
        self.numClasses = numClasses

        # build the regressor head for outputting the bounding box coordinates
        self.regressor = Sequential(
            Linear(baseModel.fc.in_features, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid()
        )

        self.classifier = Sequential(
            Linear(baseModel.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.numClasses)
        )
        self.baseModel.fc = Identity()

    def forward(self, x):
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)
        return (bboxes, classLogits)

if __name__ == '__main__':

    print("Loading dataset...")
    data = []
    labels = []
    bboxes = []
    imagePaths = []

    _image_paths = []
    for root, dirs, files in os.walk(IMAGES_PATH):
        for f in files:
            if os.path.splitext(f)[1].lower() == '.jpg':
                _image_paths.append(os.path.join(root, f))
    _label_paths = [ip.replace('image', 'label').replace('.jpg', '.txt') for ip in _image_paths]

    for lp, ip in zip(_label_paths, _image_paths[:]):
        if not osp.exists(lp):
            continue
        if not osp.exists(ip):
            continue

        image = cv2.imread(ip)  # fixme dataloader
        (h, w) = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        data.append(image)
        imagePaths.append(ip)

        with open(lp, 'r') as f:
            l_data = f.readlines()

        label = int(l_data[0])
        bbox = list(map(int, l_data[-1].split()))
        startX, startY, endX, endY = bbox
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        labels.append(label)
        bboxes.append((startX, startY, endX, endY))


    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype="float32")
    imagePaths = np.array(imagePaths)


    le = LabelEncoder()
    labels = le.fit_transform(labels)
    split = train_test_split(data, labels, bboxes, imagePaths, test_size=0.20, random_state=20022)
    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]
    (trainPaths, testPaths) = split[6:]

    (trainImages, testImages) = torch.tensor(trainImages), \
                                torch.tensor(testImages)
    (trainLabels, testLabels) = torch.tensor(trainLabels), \
                                torch.tensor(testLabels)
    (trainBBoxes, testBBoxes) = torch.tensor(trainBBoxes), \
                                torch.tensor(testBBoxes)
    # define normalization transforms
    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


    # convert NumPy arrays to PyTorch datasets
    trainDS = CarsDataset((trainImages, trainLabels, trainBBoxes),
                          transforms=transforms)
    testDS = CarsDataset((testImages, testLabels, testBBoxes),
                         transforms=transforms)
    print("[INFO] total training samples: {}...".format(len(trainDS)))
    print("[INFO] total test samples: {}...".format(len(testDS)))
    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDS) // BATCH_SIZE
    valSteps = len(testDS) // BATCH_SIZE
    # create data loaders
    trainLoader = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    testLoader = DataLoader(testDS, batch_size=BATCH_SIZE, num_workers=1)

    print("[INFO] saving testing image paths...")
    with open(TEST_PATHS, "w") as f:
        f.write("\n".join(testPaths))
    resnet = resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False


    objectDetector = SSODetector(resnet, len(le.classes_))
    objectDetector = objectDetector.to(DEVICE)
    classLossFunc = CrossEntropyLoss()
    bboxLossFunc = MSELoss()
    opt = Adam(objectDetector.parameters(), lr=INIT_LR)
    print(objectDetector)

    # initialize a dictionary to store training history
    H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
         "val_class_acc": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
        objectDetector.train()
        totalTrainLoss = 0
        totalValLoss = 0
        trainCorrect = 0
        valCorrect = 0

        for (images, labels, bboxes) in trainLoader:
            # send the input to the device
            (images, labels, bboxes) = (images.to(DEVICE),
                                        labels.to(DEVICE), bboxes.to(DEVICE))
            # perform a forward pass and calculate the training loss
            predictions = objectDetector(images)
            bboxLoss = bboxLossFunc(predictions[0], bboxes)
            classLoss = classLossFunc(predictions[1], labels)
            totalLoss = (BBOX * bboxLoss) + (LABELS * classLoss)
            opt.zero_grad()
            totalLoss.backward()
            opt.step()

            totalTrainLoss += totalLoss
            trainCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()

        with torch.no_grad():
            # set the model in evaluation mode
            objectDetector.eval()
            # loop over the validation set
            for (images, labels, bboxes) in testLoader:
                # send the input to the device
                (images, labels, bboxes) = (images.to(DEVICE),
                                            labels.to(DEVICE), bboxes.to(DEVICE))
                # make the predictions and calculate the validation loss
                predictions = objectDetector(images)
                bboxLoss = bboxLossFunc(predictions[0], bboxes)
                classLoss = classLossFunc(predictions[1], labels)
                totalLoss = (BBOX * bboxLoss) + \
                            (LABELS * classLoss)
                totalValLoss += totalLoss
                # calculate the number of correct predictions
                valCorrect += (predictions[1].argmax(1) == labels).type(
                    torch.float).sum().item()

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDS)
        valCorrect = valCorrect / len(testDS)
        # update our training history
        H["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_class_acc"].append(trainCorrect)
        H["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_class_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
            avgValLoss, valCorrect))
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime))

        # save current epoch's results
        EPOCH_PATH = os.path.sep.join([BASE_OUTPUT, f"epoch_{e + 1}"])
        create_dirs(EPOCH_PATH)
        MODEL_PATH = os.path.sep.join([EPOCH_PATH, "detector.pth"])
        LE_PATH = os.path.sep.join([EPOCH_PATH, "le.pickle"])
        print("[INFO] saving object detector model...")
        torch.save(objectDetector, MODEL_PATH)
        # serialize the label encoder to disk
        print("[INFO] saving label encoder...")
        with open(LE_PATH, "wb") as f:
            f.write(pickle.dumps(le))


    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["total_train_loss"], label="total_train_loss")
    plt.plot(H["total_val_loss"], label="total_val_loss")
    plt.plot(H["train_class_acc"], label="train_class_acc")
    plt.plot(H["val_class_acc"], label="val_class_acc")
    plt.title("Total Training Loss and Classification Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    # save the training plot
    plotPath = os.path.sep.join([PLOTS_PATH, "training.png"])
    plt.savefig(plotPath)
