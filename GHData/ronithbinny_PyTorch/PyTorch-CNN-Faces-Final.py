import os
import cv2
import numpy as np
from tqdm import tqdm
import numpy as np
import torch


# Building Data :
REBUILD_DATA = False

class Ethinicity():
    
    IMG_SIZE = 28
    indians = r'C:\Users\Ronith\Desktop\Lincode Labs\PyTorch\Faces\Indian face'
    chinese = r'C:\Users\Ronith\Desktop\Lincode Labs\PyTorch\Faces\Chinese face'
    #TESTING = "PetImages/Testing"
    LABELS = {indians: 0, chinese: 1}
    training_data = []

    indiancount = 0
    chinesecount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if f.endswith('.jpg'):
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        if label == r'C:\Users\Ronith\Desktop\Lincode Labs\PyTorch\Faces\Indian face' :
                            lab = 0
                        else :
                            lab = 1
                        self.training_data.append([np.array(img), lab])
                        #print(np.eye(2)[self.LABELS[label]])

                        if label == self.indians:
                            self.indiancount += 1
                        elif label == self.chinese:
                            self.chinesecount += 1

                    except:
                        print('NOT OK')

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        
        print('Indian : ', self.indiancount)
        print('Chinese : ', self.chinesecount)


if REBUILD_DATA:
    eth = Ethinicity()
    eth.make_training_data()
    
training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))


X = torch.tensor([i[0] for i in training_data]).view(-1, 28, 28)
X = X/255.0
y = torch.tensor([i[1] for i in training_data])

X_train = X[:-60]
y_train = y[:-60]
X_test = X[-60:]
y_test = y[-60:]

# CNN :
    
import torch.nn as nn
import torch.nn.functional as F


# calculate same padding:
# (w - k + 2*p)/s + 1 = o
# => p = (s(o-1) - w + k)/2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 28x28x1 => 28x28x8
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=8,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1) # (1(28-1) - 28 + 3) / 2 = 1
        # 28x28x8 => 14x14x8
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0) # (2(14-1) - 28 + 2) = 0                                       
        # 14x14x8 => 14x14x16
        self.conv_2 = torch.nn.Conv2d(in_channels=8,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1) # (1(14-1) - 14 + 3) / 2 = 1                 
        # 14x14x16 => 7x7x16                             
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0) # (2(7-1) - 14 + 2) = 0

        self.linear_1 = torch.nn.Linear(7*7*16, 64)
        
        self.linear_2 = torch.nn.Linear(64, 128)
        
        self.linear_3 = torch.nn.Linear(128, 128)
        
        self.linear_4 = torch.nn.Linear(128, 2)

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(out)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = F.relu(out)
        out = self.pool_2(out)
        
        out = self.linear_1(out.view(-1, 7*7*16))
        out = self.linear_2(out)
        out = self.linear_3(out)
        out = self.linear_4(out)
        
        return F.softmax(out, dim=1)


net = Net()


import torch.optim as optim

loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0001)


BATCH_SIZE = 10
EPOCHS = 50

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(X_train), BATCH_SIZE)):
        batch_X = X_train[i:i+BATCH_SIZE]
        batch_y = y_train[i:i+BATCH_SIZE]
        net.zero_grad()
        outputs = net(batch_X.view(-1, 1, 28, 28))
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(loss)
    
    
correct = 0
total = 0
pred = []
real = []

with torch.no_grad():
    for i in tqdm(range(0,len(X_test))):
        net_out = net(X_test[i].view(-1, 1, 28, 28))
        predicted_class = torch.argmax(net_out)

        if predicted_class == y_test[i]:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total, 3))


PATH = r'C:\Users\Ronith\Desktop\Lincode Labs\PyTorch\Faces\face'

torch.save(net, PATH)