import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from tqdm import tqdm

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # input, convolutional features required, window/kernel size
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        # To generate the distribution of predictions we need to
        # go from Conv2d to Linear layers

        # -- For input size of the Linear layer passing random data through Conv layers
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,2)

    def convs(self, x):
        """ A forward method for passing data through convolutional layers.
            Also serves to check the shape of the tensor at the output of the
            convolutional layers by passing random data
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        # print(x[0].shape)
        if self._to_linear is None:
            # getting the shape of the data after passing through the convolutional layers
            # x is coming as batch of data so x[0] is the first element in the batch
            # shape returns a tuple of 3 elements. First element of shape tuple * second element of shape tuple * third element of shape tuple
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        # pass data through convolutional layers
        x = self.convs(x)
        # reshape it for Linear layers
        x = x.view(-1, self._to_linear)
        # Now passing through the linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Loading the labeled data
training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))


net = Network()
print(net)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# separating out X and y from training_data
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
# Scaling the imagery. For now the pixel values are b/w 0-255 and
# these are being rescaled b/w 0 and 1.
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])
print(y[0])

# Train-Test split
split_percentage = 0.1
test_size = int(len(X)*split_percentage)
print(test_size)

train_X = X[:-test_size]
train_y = y[:-test_size]
test_X = X[-test_size:]
test_y = y[-test_size:]
print(f"Train size: {len(train_X)}")
print(f"Test size: {len(test_X)}")

BATCH_SIZE = 100
EPOCHS = 1
batch_loss = []

for epoch in range(EPOCHS):
    # start at 0, and go upto the size of train_X, step_size will be BATCH_SIZE
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        # print(i, i+BATCH_SIZE)
        # Making Batch; Reshaping for passing through the network: -1 any batch size, 1: input size, 50: dimension_of_img
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]
        # Before Fitting, Zero-ing gradients
        net.zero_grad()
        # Fitting
        outputs = net(batch_X)
        # Loss calculation
        loss = loss_function(outputs, batch_y)
        # back propagate this error
        loss.backward()
        optimizer.step()
        # saving loss value per epoch
        batch_loss.append(loss)




print(loss)
print(len(batch_loss))

# Evaluation - Test Accuracy
correct = 0
total = 0

with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        # argmax: returns the index of the maximum value in the tensor passed to it
        # test_y[0] = [1,0] and argmax will return: 0 as at 0 index we have highest value
        # The class label is one-hot vector such that on 0 index it is hot for CAT and on 1 for dog
        # and the label for CAT is 0 and for DOG is 1
        actual_class = torch.argmax(test_y[i])
        # Predict;
        # 0th element of net_out is the prediction tensor
        net_out = net(test_X[i].view(-1,1,50,50))[0]
        # print("Predicted tensor is:",net_out)

        # Applying argmax on this tensor will give us info that
        # on which index we have the highest probability value of the class
        # and the index is corresponding to the class i.e. 0=CAT 1=DOG
        predicted_class = torch.argmax(net_out)
        if predicted_class == actual_class:
            correct += 1
        total += 1


print(f"Test Accuracy is: {(correct/total)*100}%")
