import os
import cv2
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA = False

class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 conv
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)
        
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        
        x = torch.flatten(x, 1, -1)
        
        if self._to_linear is None: 
            self._to_linear = x.shape[1]
        return x
    
    def forward(self, x):
        x = self.convs(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    c_count = 0
    d_count = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            # Access the individual images inside /Cat and /Dog
            for f in tqdm(os.listdir(label)):
                try:
                    # Create the full path name here
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # eye is identity matrix. self.LABELS[label] is just dictionary and key accessing
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.c_count += 1
                    elif label == self.DOGS:
                        self.d_count += 1
                        
                except:
                    pass
                
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print(len(self.training_data))
        print("Cats: ", self.c_count)
        print("Dogs: ", self.d_count)

if REBUILD_DATA:
    dvc = DogsVSCats()
    print("Making...")
    dvc.make_training_data()
    REBUILD_DATA = False

net = Net()

training_data = np.load("training_data.npy", allow_pickle = True)

# [...][0] ==> the actual image in array format
# [...][1] ==> label

plt.imshow(training_data[1][0], cmap="gray")
print(training_data[1][1])
plt.show()

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# Testing with data from training_data
# if .Tensor, no need the , 1 after -1
x = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
x = x/255.0
y = torch.Tensor([i[1] for i in training_data])

validation_perc = 0.1
validation_size = int(len(x) * validation_perc)

# Basically 90% is for training
train_x = x[:-validation_size]
train_y = y[:-validation_size]

# 10% for testing
test_x = x[-validation_size:]
test_y = y[-validation_size:]

BATCH_SIZE = 100
EPOCHS = 3

# Training time
for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
        batch_x = train_x[i: i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i: i+BATCH_SIZE]
        
        net.zero_grad()
        
        outputs = net(batch_x)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch: {epoch}. Loss: {loss}")

# Testing for accuracy
# correct = 0
# total = 0
# with torch.no_grad():
#     for i in tqdm(range(len(test_x))):
#         real_class = torch.argmax(test_y[i])
#         net_out = net(test_x[i].view(-1, 1, 50, 50)) # returns a list, 
#         predicted_class = torch.argmax(net_out)

#         if predicted_class == real_class:
#             correct += 1
#         total += 1
# print("Accuracy: ", round(correct/total, 3))

index = int(input("Index for test image: "))
actual_img = test_x[index]

plt.imshow(actual_img, cmap="gray")
plt.show()

# torch.argmax returns the indices with the highest probability
actual_label = torch.argmax(test_y[index])
network_output = net(test_x[index].view(-1, 1, 50, 50))
predicted_label = torch.argmax(network_output)
print(network_output)
print(predicted_label)

if (predicted_label.item() == 0):
    print("Cat")
else:
    print("Dog")