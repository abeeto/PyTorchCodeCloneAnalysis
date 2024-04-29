import sys
import pymongo
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from datetime import timezone
from sklearn.model_selection import train_test_split
from torch.autograd import Variable


client = pymongo.MongoClient("mongodb://heroku_shhqhwpl:bs11g0pqvsg6r8pbtnpqed4ski@ds229068.mlab.com:29068/heroku_shhqhwpl")
database = client["heroku_shhqhwpl"]
collection = database["gestures"]

# create "x-value" tensor
x_data = []
basetime = sys.maxsize

for temp in collection.find({'label':{'$ne':None}},{'label':0,'__v':0}):
    temp['_id'] = temp['_id'].generation_time
    temp['_id'] = temp['_id'].replace(tzinfo=timezone.utc).timestamp()
    basetime = min(basetime, temp['_id'])

print(basetime)

for x in collection.find({'label':{'$ne':None}},{'_id':0,'label':0,'__v':0}):
    x_data.append(x)

xframe = pd.DataFrame(x_data)
xframe.fillna(-1)

xarrays = [np.array(df) for df in xframe.values]
x = torch.tensor(np.stack(xarrays))
x = x.type(torch.float)

# create "y-value" tensor
y_data = []
for y in collection.find({'label':{'$ne':None}}, {'_id':0,'Thumb_Value':0,'Index_Finger_Value':0,'Middle_Finger_Value':0,'Ring_Finger_Value':0,'Pinky_Finger_Value':0,'Pitch':0,'Yaw':0,'Roll':0,'__v':0}):
    if(y['label'] == 'fingerone'):
        y['label'] = 1
    if(y['label'] == 'fingertwo'):
        y['label'] = 2
    if(y['label'] == 'fingerthree'):
        y['label'] = 3
    y_data.append(y)

yframe = pd.DataFrame(y_data)
yframe.fillna(-1)

yarrays = [np.array(df) for df in yframe.values]
y = torch.tensor(np.stack(yarrays))
y = y.type(torch.float)


percent_training = 0.5
percent_testing = 1 - percent_training

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percent_testing)

x_train = Variable(x_train)
x_test = Variable(x_test)
y_train = Variable(y_train)
y_test = Variable(y_test)


class motionPredict(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(motionPredict, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = functional.relu(self.hidden(x))
        x = self.predict(x)
        return x


# set up loss function and optimizer
# exclude the id when actually doing data stuff for obvious reasons
our_model = motionPredict(n_feature=8, n_hidden=1000, n_output=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(our_model.parameters(), lr=0.001)

# training and optimization
for epoch in range(1800):
    # Forward pass: Compute predicted y by passing
    # x to the model
    pred_y = our_model(x_train)

    # Compute and print loss
    loss = criterion(pred_y, y_train)

    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

pred_y = our_model(x_test)
print("comparison of training set and predicted:")
print(y_test)
print(pred_y)