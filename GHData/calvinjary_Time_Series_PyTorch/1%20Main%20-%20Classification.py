############################## Introduction ##############################
'''
Calvin's PyTorch ML code.
Features: 
    1. full parallel CPU and GPU support, dynamically chosen at the start of the code
    2. full batch training support, chosen as a hyperparameter by the user
    3. full support for training and testing loader, allowing chunks to be loaded and tested
    4. support for any number of features, as well as any prediction window size, and any sequence length

Comments about Variable names:
    "Train" means the data used for training (the ground truth, the data known and used to train the model). "Test" means the unknown data not used to train the model that we used to test the model's accuracy
    "x" implies the input data (meaning the feature information, the input data to the model) and "Y" implies the label (the ground truth, the value to the predicted, the output of the model)
    
In the context of the inner workings of the RNN/GRU/LSTM:
    X = input
    h = hidden
    o = output
    xh = input-to-hidden
    hh = hidden-to-hidden
    h0 = initial hidden state
    
    
'''
############################## IMPORTS ##############################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable  #variables are a wrapper around a tensor that allows it to accumulate gradients 
# any variable you want to optimize later is requires_grad = True

torch.manual_seed(42)  # this is so the model is deterministic and gives the same results every time


# Code to activate GPU usage. Remember only the model and the variables need to be explicitly passed to the GPU in the code  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # one line version
if torch.cuda.is_available():
    device =  torch.device("cuda")    # set the PyTorch Device to the first CUDA device detected. If this breaks change it from "cuda:0" to "cuda"
    print('Torch CUDA device is: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    #x = torch.cuda.memory_reserved() / 1024**2  # converting bytes to megabytes
    #print('total CUDA memory reserved is:', x, 'MB')  
    torch.cuda.empty_cache()
    #learn.destroy()

else:
    device = torch.device("cpu")
    print('CUDA not detected, instead using:', device)

############################################## DATASET ####################################

try:
    pdDataTableOriginal1 = pd.read_csv('E:/Dropbox/3 PhD Research/Project 1/Day-3-mobility.txt', header = 0, engine='python')
except:
    pdDataTableOriginal1 = pd.read_csv('C:/Dropbox/3 PhD Research/Project 1/Day-3-mobility.txt', header = 0, engine='python')
# truncating to get rid of all the zeroes
pdDataTable1 = pdDataTableOriginal1.truncate(before=45266)     # for full dataset with no 0's you want: (before=14114)
# re-indexing
pdDataTable1 = pdDataTable1.reset_index(drop=True)
##### Optional Truncation for not decompressing
pdDataTable1 = pdDataTable1.drop('n', axis="columns")  
y = pdDataTable1.iloc[:,:].values.astype(int)  # [0,:]
#x1 = x.values.astype(int)

#X_Test_set = y[len(y) - test_samples:,:] # starts 40 points from the end and goes all the way to the end
#X_Train_set = y[:(len(y) - test_samples),:]  # start from the beginning all the way to the start of the test size 



'''
# working data
X_Train_set = [[1,0,1,0,1,0,1],(1,0,1,0,1,0,1),(1,0,1,0,1,0,1),(1,0,1,0,1,0,1)]
X_Test_set  = [(0,1,0,1,0,1,0),(1,0,1,0,1,0,1),(0,1,0,1,0,1,0),(1,0,1,0,1,0,1)]
'''


# data should be in the 3D array of: samples(depth) X sequence length (coloumns) X features (rows)



X_Train_set =  [[[1,  0,  1,  0,  1,  0,  1],
                 [1,  0,  1,  0,  1,  0,  1],
                 [1,  0,  1,  0,  1,  0,  1]],

                [[1,  0,  1,  0,  1,  0,  1],
                 [1,  0,  1,  0,  1,  0,  1],
                 [1,  0,  1,  0,  1,  0,  1]]]


X_Test_set =   [[[1,  0,  1,  0,  1,  0,  1],
                 [1,  0,  1,  0,  1,  0,  1],
                 [1,  0,  1,  0,  1,  0,  1]],

                [[1,  0,  1,  0,  1,  0,  1],
                 [1,  0,  1,  0,  1,  0,  1],
                 [1,  0,  1,  0,  1,  0,  1]]]








'''
X_Train_set =  [[[1,  2,  3,  4,  5,  6,  7],
                [11, 12, 13, 14, 15, 16, 17],
                [21, 22, 23, 24, 25, 26, 27]],

               [[31, 32, 33, 34, 35, 36, 37],
                [41, 42, 43, 44, 45, 46, 47],
                [51, 52, 53, 54, 55, 56, 57]]]


X_Test_set =  [[[1,  2,  3,  4,  5,  6,  7],
                [11, 12, 13, 14, 15, 16, 17],
                [21, 22, 23, 24, 25, 26, 27]],

               [[31, 32, 33, 34, 35, 36, 37],
                [41, 42, 43, 44, 45, 46, 47],
                [51, 52, 53, 54, 55, 56, 57]]]
'''




#X_Test_set  = [(1,0,1,0,1,0,1),(1,0,1,0,1,0,1),(1,0,1,0,1,0,1),(1,0,1,0,1,0,1)]
#X_Test_set  = [(0,1,0,1,0,1,0),(1,0,1,0,1,0,1),(0,1,0,1,0,1,0),(1,0,1,0,1,0,1)]



########################################## SETUP TRAIN AND TEST SETS ####################################

# Data Hyperparameters
Sequence_Length = 4  # essentially the length of the moving window of input data
#test_samples = 4

# Model Hyperparameters

Input_Features = len(X_Train_set[0])   # Input_Features is the number of expected features (different characteristics per individual time interval) in the input. 1 data point per timestamp is input_size of 1
Hidden_Size = 50    # Hidden_Size: The number of neurons in the hidden state h.
Number_Layers = 1    # Number_Layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.    
Number_Outputs = 2   # output_size is how many outputs you're predicting. For classification; it's the number of unique classes. For regression; it's how many numbers out into the future

# Training Hyperparameters
Batch_Size = 2       # number of chunks to split all the data into. example 10 means train with 10% of the data at one time. Too low is inefficient. too high is slow and memory intensive
Number_Epochs = 50   # 10 epochs means running through the whole dataset (all the data) 10 times

########################################## PREPARE THE TRAINING DATA ##############################


# make the training and testing data a numpy array
np_X_Train_set = np.array(X_Train_set, dtype = np.int_) #  np.float32    np.int_
np_X_Test_set = np.array(X_Test_set, dtype = np.int_)


# make the training and testing data a torch tensor, automatically loaded to either the GPU or the CPU
X_Train = torch.LongTensor(np_X_Train_set).to(device)   # can use FloatTensoror IntTensor
X_Test = torch.LongTensor(np_X_Test_set).to(device)  #  can use FloatTensoror IntTensor
print(X_Train[0].cpu().numpy())   # shows one full sequence of data


# debugging stuff



# Processing the data into windows and labels


def SequentialData(AllDataInput, TrainingDataLength): # input is all data, desired training sequence window size
    Output = []  
    for CurrentSample in range(len(AllDataInput)):   # iterating through all of the provided user data samples
        TrainingDataInput = AllDataInput[CurrentSample]  # focusing on one user data sample at a time
        for i in range(len(TrainingDataInput[0])-TrainingDataLength):  # i is the number of unique window frames + predictions within one sequence
                TrainingData = TrainingDataInput[:,i:i+TrainingDataLength]  # this is the training data for a given window size. It's all rows (these are all features), and then a number of coloumns equal to the window size
                OutputLabel = TrainingDataInput[0,i+TrainingDataLength]  # OutputLabel is the final predicted unique sequence, it sits on the top row which is the data to be predicted row
                Output.append((TrainingData, OutputLabel))               # this combines the training window, predicted final sequeence
    return Output
    

# Calling the processing method SequentialData. the data is a tuple where first element is data and the second is the label (prediction)
X_TestProcessed = SequentialData(X_Test, Sequence_Length)
X_TrainProcessed = SequentialData(X_Train, Sequence_Length)
X_TrainProcessed_Original = X_TrainProcessed




print((X_TrainProcessed[0]))  # shows one data instance (one window and one prediction)


# Loading our data into the Torch DataLoader
Train_Loader = torch.utils.data.DataLoader(dataset = X_TrainProcessed, batch_size = Batch_Size, shuffle = False)   # pin_memory = True is for CUDA only
Test_Loader  = torch.utils.data.DataLoader(dataset = X_TestProcessed,  batch_size = Batch_Size, shuffle = False)

'''
for Input ,_ in Train_Loader:    # this shows the first batch. remove the  ,_  to see all batches
    print(Input[0].cpu().numpy()) # shows full contents of a data loader
    break  # remove this break to show the full contents of a batch, not just the first element
'''  
# requires_grad = True , creates the computational graph, necessary to calculate the gradients. example weights = torch.ones(4, requires_grad = True)


#################################### CALCULATING PARAMETERS BASED ON TRAINING DATA ##############################


# Number of iterations is how many runs through the training part of the model. It is calculated as the number of epochs * total number of data samples / batch size
Number_Iterations = int(Number_Epochs * len(X_TrainProcessed) / Batch_Size)

# Calculating how many Iterations of the model would be considered 5% training progress completion
Five_Percent_Progress = int(Number_Iterations / 20)


################################################ DEFINE THE MODEL ##############################

class Model_Class(nn.Module):

    def __init__(self, Input_Features, Number_Outputs, Hidden_Size, Number_Layers):  # here we are declaring / defining the methods and algorithms we will be using later
        
        super(Model_Class, self).__init__()   # super function is called to inherit everything from nn.Module   super().__init__()  

        # Add an LSTM layer: (the sequence)
        #self.lstm = nn.LSTM(Input_Features, Hidden_Size, Number_Layers)  I think input_size is number of features per timestamp 
        
        self.rnn = nn.RNN(Input_Features, Hidden_Size, Number_Layers, batch_first = True, nonlinearity = 'relu') 
   
        self.linear = nn.Linear(Hidden_Size, Number_Outputs)         
        
        # Add a fully-connected layer. from 50 neurons down to 1 neuron for prediction
        # self.linear = nn.Linear(input features(hidden_size), output features (output_size))
        # self.linear = nn.Linear(30, 1) # this is for regression only. takes in the hidden_size and reduces it down to the output_size
        # self.linear = nn.Linear(hidden_size, output_size)  


        # This is the sequence, along with a tuple that is the hidden and cell state
        # Initialize the hidden state h0 and the cell state c0:
        #self.hidden = (torch.zeros(num_layers, batch_size, self.hidden_size),  #hidden state h0 short-term memory
        #               torch.zeros(num_layers, batch_size, self.hidden_size))  #cell state c0 long-term memory
        
    def forward(self, x):   #  X is Model_Input used in this code. This is the using of the algorithms used in the ML model
             
        # format of the hidden layer for an RNN is: (number of layers, Batch_Size (could be samples as X.size(0)), hidden size)
        
        HiddenState = torch.zeros(Number_Layers, Batch_Size, Hidden_Size).to(device)
        #HiddenState = HiddenState.long()
        #model.hidden = (torch.zeros(num_layers,batch_size,model.hidden_size),  # hidden state h0 short-term memory
        #                torch.zeros(num_layers,batch_size,model.hidden_size))  # cell state c0 long-term memory
             
        # get RNN unit output
        # Output is of dimensions: (batch size (or sample), time step, feature)
        # hn is of dimenensions: batch size (or sample), layer, feature
        #x = x.type(torch.LongTensor)
        #HiddenState = HiddenState.type(torch.LongTensor)
        
        Output, hn = self.rnn(x.float(), HiddenState)  # .float()    .long()
    
        # We only want hidden state for final time step for our use. The output here has 3 dimensions of shape: Batch_Size (could be samples as X.size(0), Sequence_Length, Hidden_Size 
        # calling the linear layer, we want hidden(sequence length) at the final time step. Samples * hidden size -> samples * outputs(classes)
        Output = self.linear(Output[:, -1, :])  # [: (grab all samples in the first dimension), -1 (grab only the last item in the second dimension), : (grab all the features in the final dimension)]      
        # the -1 term here is for a final output type oF RNN
        # Output = self.linear(Output) # this is the code to get all the outputs for an RNN that takes an output for all time points rather than just the final one 
    
        #softmax = torch.nn.Softmax(dim=1) 
        #Output = softmax(Output)
    
        return Output


#################################### INSTANTIATE THE MODEL, DEFINE LOSS FUNCTION & OPTMIZATION FUNCTION ####################################

Main_Model = Model_Class(Input_Features, Number_Outputs, Hidden_Size, Number_Layers).to(device) 


print(Main_Model)

print(Main_Model.parameters)
#print(list(model.parameters())[0].size())


# loss function here calculates how far the predicted values are from the training values (used for backpropagation and upgrading the network)
#criterion = nn.MSELoss()     # this is to compare single values   this is only for regression
criterion = nn.CrossEntropyLoss()     # the cross entropy loss computes softmax as well. cross entropy loss also does not require one-hot-encode of features

# this is the optimzer to use, can be torch.optim.SGD or also torch.optim.Adam
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   
optimizer = torch.optim.SGD(Main_Model.parameters(), lr=0.01, momentum=0.5)  

# to view the parameters of the model run:
print('Details of the Model are: ', Main_Model)

# this function simply allows the user to count the number of parameters in the model
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')
    
count_parameters(Main_Model)  # this outputs the number of parameters


########################################## TRAIN THE MODEL ####################################

classDict = {   
    "13400" : 0,
    "13401" : 1,
    "13402" : 2,
    }

start_time = time.time()

#num_total_steps = len(train_loader)


Iterations = int(0)

loss_list = []
iteration_list = []
accuracy_list = []

# calvin stuff
'''
test_loss = []
train_loss = []
test_outputs = []
test_criterion = []
'''


x = torch.cuda.memory_reserved() / 1024**2  # converting bytes to megabytes
print('total CUDA memory reserved is:', x, 'MB')  

for epoch in range(Number_Epochs):

    for i, (X_TrainProcessed, Y_TrainProcessed) in enumerate(Train_Loader):
        Main_Model.train() # set the model to training mode
        
        # clears (zeroes) the gradients with respect to the parameters & hidden states (don't want to accumulate gradients after every epoch)
        optimizer.zero_grad()

        # batch size is number of samples we send to the model at one time, should be powers of 2
        # sequence length is length of the sequence of input data
        # input dimensions is number of features (dimensions) in the dataset

        #with torch.no_grad():
            
        #Batch_Size = Batch_Size.type(torch.LongTensor)
        #Sequence_Length = Sequence_Length.type(torch.LongTensor)
        #Input_Features = Input_Features.type(torch.LongTensor)
        
            
        Model_Input_Train = X_TrainProcessed.view(Batch_Size, Sequence_Length, Input_Features).to(device)    # can use reshape
        
        # Forward Propagation (Forward Pass) to get the model's generated output (calculating new outputs with our current model)
        Y_Pred_Train = Main_Model(Model_Input_Train).to(device)
        # output from this model is of shape: (Batch_Size, Number_Outputs)

        #Y_Train = torch.IntTensor(Y_TrainProcessed).to(device)
        #Y_Pred = torch.IntTensor(Y_Pred).to(device)        
    
        # Calculate the loss function. It's the distance between model calculated predicted values (Y_Pred) and true ground truth label values (Y_Train)
        Train_Loss = criterion(Y_Pred_Train, Y_TrainProcessed)   # first term is model output values, second term is known ground truth labels
        
        # calculates the new gradients with respect to the backward pass, for this epoch
        Train_Loss.backward()
        
        # updating model parameters based on this new epoch 
        optimizer.step() 
        #train_loss.append(loss.item())
        
        #torch.cuda.empty_cache()
        
        # Calculate Accuracy 
        if (Iterations % Five_Percent_Progress == 0) or (Iterations+1 == Number_Iterations):
            Main_Model.eval()     # put the model into evaluation mode
            with torch.no_grad():  # disables backpropagdation which reduces memory usage
                correct, total, accuracy = float(0), float(0), float(0)
    
                # Iterate through test dataset
                for (X_TestProcessed, Y_TestProcessed) in Test_Loader:
                    #images = Variable(images.view(-1, seq_dim, input_dim))
                    #with torch.no_grad():
                    Model_Input_Test = X_TestProcessed.view(Batch_Size, Sequence_Length, Input_Features).to(device)
                        
                    Y_Pred_Test = Main_Model(Model_Input_Test).to(device)
                    
                    #test_outputs.append(Y_Pred_Test) 
                    Test_Loss = (criterion(Y_Pred_Test, Y_TestProcessed))
                    #test_loss.append(criterion(Y_Pred_Test, Y_TestProcessed))
                                   
                    # Get predictions from the maximum value
                    predicted = torch.max(Y_Pred_Test.data,1)[1]
                    
                    # Total number of labels
                    total += Y_TestProcessed.size(0)
                    
                    correct += (predicted == Y_TestProcessed).sum()
                
                accuracy = 100 * correct / total
                
                
                #torch.cuda.empty_cache()
                
                # store loss and iteration
                '''
                loss_list.append(loss.data)   # can do either loss.data or loss.item()
                iteration_list.append(iterations)
                accuracy_list.append(accuracy)
                '''
                
                if Iterations+1 == Number_Iterations: # ensures the epoch number and iteration number line up for the very final iteration
                    epoch += 1
                    Iterations += 1
                
                #if iterations % 10 == 0:
                # Print Loss
                print('Epoch: {:4.0f} | Iteration: {:5.0f} | Train Loss: {:3.6f} | Test Loss: {:3.6f} | Correct: {:2.0f} | Total: {:2.0f} | Accuracy: {:4.2f} %'.format(epoch, Iterations, Train_Loss, Test_Loss, correct, total, accuracy))

        Iterations += 1

x = torch.cuda.memory_reserved() / 1024**2  # converting bytes to megabytes
print('total CUDA memory reserved is:', x, 'MB') 






#print(f'\nDuration: {time.time() - start_time:.0f} seconds')

# total_time = time.time() - start_time
# print(total_time/60)




################## EVALUATE THE MODEL #####################








# total_time = time.time() - start_time
# print(total_time/60)

#torch.save(model.state_dict(), ' my model.pt')






