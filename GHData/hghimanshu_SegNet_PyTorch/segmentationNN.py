import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import pickle
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

trainDataset = pickle.load(open("/home/techject/Desktop/Himanshu/Codes/MLND-Capstone-master/full_CNN_train.p", "rb" ))
labels = pickle.load(open("/home/techject/Desktop/Himanshu/Codes/MLND-Capstone-master/full_CNN_labels.p", "rb" ))

# Make into arrays as the neural network wants these
trainDataset_np = np.array(trainDataset)
labels_np = np.array(labels)

# Normalize labels - training images get normalized to start in the network
labels_np = labels_np / 255
# Shuffle images along with their labels, then split into training/validation sets
train_images, labels = shuffle(trainDataset_np, labels_np)
# Test size may be 10% or 20%
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(train_images, labels, test_size=0.1)
#Converting them into Tensors
#if torch.cuda.is_available():
#    trainDataset = torch.from_numpy(trainDataset_np)
#    labels = torch.from_numpy(labels_np)
X_train = torch.from_numpy(X_train_np)
X_test = torch.from_numpy(X_val_np)
Y_train = torch.from_numpy(y_train_np)
Y_test = torch.from_numpy(y_val_np)

#else:
#    print("You need a GPU Congiguration to run the script!!")

X_train, X_test, Y_train, Y_test = Variable(X_train), Variable(X_test), Variable(Y_train), Variable(Y_test)

batch_size = 128
epochs = 10

new_x = []
for i in range(0,128):
    new_x.append(X_train[i])
    
X_train_new=np.asarray(new_x)

input_shape = X_train_new.shape

MODEL_STORE_PATH = '/home/techject/Desktop/Himanshu/Codes/segnet pytorch/model/'

if not os.path.exists(MODEL_STORE_PATH):
    os.makedirs(MODEL_STORE_PATH)



#### Making the Network ############
#in_channels = 3 as the image is colored
class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
	#Conv layers
        self.batchNorm = nn.modules.BatchNorm2d(input_shape)
        self.Convlayer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU())
        self.Convlayer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU())
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Convlayer3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.Dropout(0.2))
        self.Convlayer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.Dropout(0.2))
        self.Convlayer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.Dropout(0.2))
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Convlayer6 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.Dropout(0.2))
        self.Convlayer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.Dropout(0.2))
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
	#Deconv layers
        self.upSample1 = nn.UpsamplingBilinear2d(size=(2,2))
        self.DeConvlayer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.Dropout(0.2))
        self.DeConvlayer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.Dropout(0.2))
        self.upSample2 = nn.UpsamplingBilinear2d(size=(2,2))
        self.DeConvlayer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.Dropout(0.2))
        self.DeConvlayer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.Dropout(0.2))
        self.DeConvlayer5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), nn.Dropout(0.2))
        self.upSample3 = nn.UpsamplingBilinear2d(size=(2,2))
        self.DeConvlayer6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU())
        self.DeConvlayer7 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=2),
            nn.ReLU())
      #forward propagation function
    def forward(self, x):
        out = self.batchNorm(x)
        out = self.Convlayer1(out)
        out = self.Convlayer2(out)
        out = self.maxPool1(out)
        out = self.Convlayer3(out)
        out = self.Convlayer4(out)
        out = self.Convlayer5(out)
        out = self.maxPool2(out)
        out = self.Convlayer6(out)
        out = self.Convlayer7(out)
        out = self.maxPool3(out)
        out = self.upSample1(out)
        out = self.DeConvlayer1(out)
        out = self.DeConvlayer2(out)
        out = self.upSample2(out)
        out = self.DeConvlayer3(out)
        out = self.DeConvlayer4(out)
        out = self.DeConvlayer5(out)
        out = self.upSample3(out)
        out = self.DeConvlayer6(out)
        out = self.DeConvlayer7(out)
        
        return out

model = SegNet()
#
#if torch.cuda.is_available():
#    model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

total_step = X_train.shape[0]
loss_list = []
acc_list = []
for epoch in range(epochs):
    for i, (images, labels) in enumerate(zip(X_train, Y_train)):
        
        images = images.unsqueeze(0)
        labels = labels.unsqueeze(0)
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in zip(X_test, Y_test):
        images = images.unsqueeze_(0)
        labels = labels.unsqueeze_(0)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model : {} %'.format((correct / total) * 100))

# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'seg_net_model.pth')
