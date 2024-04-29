#import the libraries
import pandas as pd
import numpy as np
import torch as torch  
from sklearn.cross_validation import train_test_split
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

LOCATION_OF_DATA_FOR_TRAINING = 'time_taken_home_to_office - data.csv'

X = pd.read_csv(LOCATION_OF_DATA_FOR_TRAINING, header=0, usecols=['day','start_time', 'is_any_road_blocked_in_between', 'mode_of_transport'])

# transform all coloumns in X to numerical data and one-hot encode
# We don't need feature scaling as the code below takes care of it as well
X = pd.get_dummies(X, columns=['day','start_time', 'is_any_road_blocked_in_between', 'mode_of_transport']).values

Y = pd.read_csv(LOCATION_OF_DATA_FOR_TRAINING, header=0, usecols=['time_taken']).values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# get number of columns in training data
# which means how many variables you have for training
n_cols = X_train.shape[1]


# -----------------------------------------------------------------------------
# Class for making the Neural Network

# Make a regressor with three hidden layers
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(n_cols, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 1)

        #self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x
 
    
# -----------------------------------------------------------------------------
# Let's train the model
    
train_batch = np.array_split(X_train, 10)
label_batch = np.array_split(y_train, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(len(train_batch)):
    train_batch[i] = torch.from_numpy(train_batch[i]).float()
for i in range(len(label_batch)):
    label_batch[i] = torch.from_numpy(label_batch[i]).float().view(-1, 1)

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float().view(-1, 1)

model = Regressor()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=.01)
epochs = 300

train_losses, test_losses = [], []
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for i in range(len(train_batch)):
        optimizer.zero_grad()
        output = model(train_batch[i])
        loss = torch.sqrt(criterion(torch.log(output), torch.log(label_batch[i])))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    else:
        test_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            model.eval()
            predictions = model(X_test)
            test_loss += torch.sqrt(criterion(torch.log(predictions), torch.log(y_test)))
                
        train_losses.append(train_loss/len(train_batch))
        test_losses.append(test_loss)

        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(train_batch)),
              "Model Accuracy: {:.3f}.. ".format((1-test_loss)*100))
        
        
# Plot the loss on graph       
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)   


#------------------------------------------------------------------------------    
# let's do the prediction

# input for if time is 8.30 AM on Monday
inpt1 = np.array([[1,0,0,0,0,1,0,0,0,0,0,0,1,0]])
# input for if time is 9.00 AM on Monday
inpt2 = np.array([[1,0,0,0,0,0,1,0,0,0,0,0,1,0]])
# input for if time is 9.30 AM on Monday
inpt3 = np.array([[1,0,0,0,0,0,0,1,0,0,0,0,1,0]])
# input for if time is 10.00 AM on Monday
inpt4 = np.array([[1,0,0,0,0,0,0,0,1,0,0,0,1,0]])
# input for if time is 10.30 AM on Monday
inpt5 = np.array([[1,0,0,0,0,0,0,0,0,1,0,0,1,0]])

#Convert numpy array to tensor because PyTorch works on Tensors
tensor1 = torch.Tensor(inpt1)
y_pred1 = model(tensor1)

tensor2 = torch.Tensor(inpt2)
y_pred2 = model(tensor2)

tensor3 = torch.Tensor(inpt3)
y_pred3 = model(tensor3)

tensor4 = torch.Tensor(inpt4)
y_pred4 = model(tensor4)

tensor5 = torch.Tensor(inpt5)
y_pred5 = model(tensor5)

best_time = min(y_pred1,y_pred2,y_pred3,y_pred4, y_pred5)

if y_pred1 is best_time:
    print "Best time to start for office on Monday is 8.30 AM, it takes ", best_time[0].item() , ' minutes'

elif y_pred2 is best_time:
    print 'Best time to start for office on Monday is 9.00 AM, it takes ', best_time[0].item() , ' minutes'
    
elif y_pred3 is best_time:
    print 'Best time to start for office on Monday is 9.30 AM, it takes ', best_time[0].item() , ' minutes'

elif y_pred4 is best_time:
    print 'Best time to start for office on Monday is 10.00 AM, it takes ', best_time[0].item() , ' minutes'
    
elif y_pred5 is best_time:
    print 'Best time to start for office on Monday is 10.30 AM, it takes ' , best_time[0].item() , ' minutes' 
