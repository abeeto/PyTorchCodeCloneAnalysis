#!/usr/bin/env python
# coding: utf-8

class CNN(nn.Module):
    
    def __init__(self, num_classes=input_features):
        super(CNN, self).__init__()
       
        # RGB images, input channels = 3. 
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # Max-pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # A drop layer can be added here for improving regression performance to help prevent overfitting
        #self.drop = nn.Dropout2d(p=0.2)
        
        # Flatten in order to feed them to a fully-connected layer, output_feature=1 for regression problems
        self.fc = nn.Linear(in_features= 40*40*256, out_features=1)

    def forward(self, x):

        # Use the ReLU activation function after each hidden layer (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x))) 
        x = F.relu(self.pool(self.conv2(x))) 
        x = F.relu(self.pool(self.conv3(x))) 
        x = F.relu(self.pool(self.conv4(x)))  
        x = F.relu(self.pool(self.conv5(x))) 
        
        # Select some features to drop to prevent overfitting (only drop during training)
        #x = F.dropout(self.drop(x), training=self.training)
        
        # Flatten
        # x = x.view(-1, 25 * 25 * 24)
        x = torch.flatten(x, 1)
        # Feed to fully-connected layer to predict
        x = self.fc(x)
        # Return class probabilities via a softmax function 
        return torch.softmax(x, dim=1)

