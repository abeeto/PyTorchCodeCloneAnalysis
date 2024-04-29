# import the necessary packages
from cProfile import label
import Config
from operator import le
from tracemalloc import start
from ObjectDetector import objectDetector
from Custom_tensor_dataset import CustomTensorDataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import time
import cv2
import os

# ---------------------------------------------------------------- # 

# Initialize the listt of data (images), class labels, target bounding 
# box coordinates, and image paths

print("[INFO] loading dataset ....")

data = []
labels = []
bboxes = []
imagePaths = []

# loop over all CSV files in the annotation directory

for csvPath in paths.list_files(Config.ANNOTS_PATH, validExts=(".csv")):  
    # load the contents of the current CSV 
    rows = open(csvPath).read().strip().split("\n")

    for row in rows:
        row = row.split(",")
        (filename,startX, startY, endX, endY, label_) = row

        # derive the path to the input image, load the image (in OpenCV format), and grab its dimensions 
        imagePath = os.path.join(Config.IMAGE_PATH,label_, filename)
        image = cv2.imread(imagePath)
        (imageHeight, imageWidth) = image.shape[0:2]

        # Scale the bouding box coordinates relative to the spatial 
        # dimension of the input image 

        startX = float(startX) / imageWidth
        startY = float(startY) / imageHeight
        endX = float(endX) / imageWidth
        endY = float(endY) / imageHeight


        # load the image and preprocess it 
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224))

        # Update our list of data, class labels, bouding box, and 
        # image paths

        data.append(image)
        labels.append(label_)
        bboxes.append((startX, startY, endX, endY))
        imagePaths.append(imagePath)





# convert the data, class labels, bounding boxes, and image paths to
# NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

# Partition the data into trainig and testing splits using 80% of
# the data for training and the reaming 20% for testing 

split = train_test_split(data,labels,bboxes,imagePaths,test_size = 0.20, random_state=42)

# unpack the data slipt 

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]


# Convert Numpy arrays to Pytorch Tensors    
(trainImages, TestImages)  = torch.tensor(trainImages),torch.tensor(testImages)
(trainLabels, TestLabels)  = torch.tensor(trainLabels),torch.tensor(testImages)
(trainBBoxes, testBBoxes)  = torch.tensor(trainBBoxes),torch.tensor(testBBoxes)


# define normalization transforms

transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(mean=Config.MEAN, std = Config.STD) ])


# Convert NumPy Array to Pytorch datasets 

trainDS = CustomTensorDataset((trainImages,trainLabels,trainBBoxes) , transforms=transforms)

testDS = CustomTensorDataset((testImages,testLabels,testBBoxes) , transforms=transforms)

print("[INFO] total training samples: {}...".format(len(trainDS)))
print("[INFO] total test samples: {}...".format(len(testDS)))

# calculate steps per epoch for training and validation set  
trainSteps = len(trainDS) // Config.BATCH_SIZE
valSteps = len(testDS) // Config.BATCH_SIZE


trainLoader = DataLoader(trainDS, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), pin_memory=Config.PIN_MEMORY)
testLoader = DataLoader(testDS, batch_size=Config.BATCH_SIZE,num_workers=os.cpu_count(), pin_memory=Config.PIN_MEMORY)


# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
f = open(Config.TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()

# load the ResNet50 network
resnet = resnet50(pretrained=True)
# freeze all ResNet50 layers so they will *not* be updated during the
# training process
for param in resnet.parameters():
	param.requires_grad = False

objectDetector = objectDetector(resnet,len(le.classes_))
objectDetector = objectDetector.to(Config.DEVICE)

# define our loss function 

classLossFunc = CrossEntropyLoss()
bboxesLossFunc = MSELoss()

# initialize the optimzer, compile the model, and show the model 
opt = Adam(objectDetector.parameters(), lr=Config.INIT_LR)
print(objectDetector)

# initialize a dictionary to store training history
H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
	 "val_class_acc": []}


# loop over epochs 


startTime = time.time()

for epoch in tqdm(range(Config.NUM_EPOCHS)):

    print("[INFO] epoch: {}".format(epoch+1))

    objectDetector.train()

    # reset the loss counters
    total_train_loss = 0
    total_val_loss = 0

    # Loop over training set 
    for (images,labels,bboxes) in trainLoader:
        # send the input to the divece 
        (images,labels, bboxes) = ((images.to(Config.DEVICE)), labels.to(Config.DEVICE), bboxes.to(Config.DEVICE))


        # perfome a forward pass and calculate the training loss 
        predictions = objectDetector(images)
        bboxesLoss = bboxesLossFunc(predictions[0], bboxes)
        classLoss = classLossFunc(predictions[1], labels)
        totalLoss = (Config.BBOX * bboxesLoss) + (Config.LABELS * classLoss)

        # reset the gradients, perform backpropagation step, 
        # and uptdate the weights 

        opt.zero_grad()
        totalLoss.backward()
        opt.step()

        totalTrainLoss = totalLoss + totalTrainLoss
        trainCorrect = (predictions[1].argmax(1) == labels).type(torch.float).sum().item() + trainCorrect
    
    
    with torch.no_grad():
        # putting the in model in evaluation mode
        objectDetector.eval()

        # loop over the validation set 
        for (images,labels,bboxes) in testLoader:
            # send the input to the divece 
            (images,labels, bboxes) = ((images.to(Config.DEVICE)), labels.to(Config.DEVICE), bboxes.to(Config.DEVICE))

            # make the predictions and calculate the validation loss 
            predictions = objectDetector(images)
            bboxesLoss = bboxesLossFunc(predictions[0], bboxes)
            classLoss = classLossFunc(predictions[1], labels)

            totalLoss = (Config.BBOX * bboxesLoss) + (Config.LABELS * classLoss) + totalLoss

            totalValLoss = totalLoss + totalValLoss

            # calculate the number of correct predictions  
            valCorrect = (predictions[1].argmax(1) == labels).type(torch.float).sum().item() + valCorrect


    # calculate the average training and validation loss 
    avgTrainLoss = totalTrainLoss / trainSteps 
    avgValLoss = totalValLoss / valSteps


    # calculate the training and validation accuracy 
    avgTrainAcc = trainCorrect / len(trainDS)
    avgValAcc = valCorrect / len(testDS)

    # uptdate our training history 
    H["total_train_loss"].append(totalTrainLoss.cpu().detach().numpy())
    H["train_class_acc"].append(trainCorrect)
    H["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_class_acc"].append(valCorrect)


    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(Config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect)) 
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))



# serialize the model to disk 
print("[INFO] saving object detector model...")
torch.save(objectDetector, Config.MODEL_PATH)

# serialize the label enconder to disk 
print("[INFO] saving label encoder...")
f = open(Config.LE_PATH, "wb")
f.write(pickle.dump(le))
f.close()

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
plotPath = os.path.sep.join([Config.PLOTS_PATH, "training.png"])
plt.savefig(plotPath)