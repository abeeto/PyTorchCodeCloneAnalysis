# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:12:09 2018

@author: vprayagala2

Sentiment Analysis on Text
"""
#%%
import numpy as np

from string import punctuation
from collections import Counter
from scipy import stats

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
#%%
#Load the Data
# read data from text files
with open("C:\\data\\SentimentTrain\\reviews.txt", 'r') as f:
    reviews = f.read()
with open("C:\\data\\SentimentTrain\\labels.txt", 'r') as f:
    labels = f.read()
#%%
print(punctuation)

# get rid of punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])


# split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()

print("Word Vocabulary Length:%d"%(len(words)))
#%%
#Encode the words into integers
## Build a dictionary that maps words to integers
word_to_int_dict = None

## use the dict to tokenize each review in reviews_split
counts=Counter(words)
vocab=sorted(counts, key=counts.get, reverse=True)
word_to_int_dict = {word : i for i,word in enumerate(vocab,1)}
## store the tokenized reviews in reviews_ints
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([word_to_int_dict[word] for word in review.split()])
#%%
# stats about vocabulary
print('Unique words: ', len((word_to_int_dict)))  # should ~ 74000+
# print tokens in first review
print('Tokenized review: \n', reviews_ints[:1])
#%%
# 1=positive, 0=negative label conversion
labels = labels.split("\n")
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels])
print("Number of Labels:%d"%(len(encoded_labels)))
#%%
# outlier review stats
review_len_list = [len(x) for x in reviews_ints]
review_lens = Counter(review_len_list)

min_len = min(review_len_list)
avg_len = int(np.mean(review_len_list))
max_len = max(review_len_list)

print("Minimum length of a review: {}".format(min_len))
print("Average-length of review: {}".format(avg_len))
print("Maximum-length of review: {}".format(max_len))

print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(review_lens[max_len]))
#%%
non_zero_idx = [ii for ii,review  in enumerate(reviews_ints) if len(review) != 0]

reviews_ints = [reviews_ints[i] for i in non_zero_idx]
encoded_labels = np.array([encoded_labels[i] for i in non_zero_idx])
print('Number of reviews before removing outliers: ', len(reviews_ints))
#%%
def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    ## implement function
    
    features=np.zeros((len(reviews_ints), seq_length), dtype=int)
    
    for i, review in enumerate(reviews_ints):
        features[i,-len(review):] = np.array(review)[:seq_length]
        
    return features

# Test your implementation!

seq_length = 200

features = pad_features(reviews_ints, seq_length=seq_length)

## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches 
print(features[:30,:10])
#%%
split_frac = 0.8
split_idx = int(len(features)*split_frac)
## split data into training, validation, and test data (features and labels, x and y)
train_x , remaining_x = features[:split_idx] , features[split_idx:]
train_y , remaining_y = encoded_labels[:split_idx] , encoded_labels[split_idx:]

split_test_idx = int(len(remaining_x)*0.5)
val_x , test_x = remaining_x[:split_test_idx] , remaining_x[split_test_idx:]
val_y , test_y = remaining_y[:split_test_idx] , remaining_y[split_test_idx:]
## print out the shapes of your resultant feature data
print("Train Data Shape:{}".format(train_x.shape))
print("Validation Data Shape:{}".format(val_x.shape))
print("Test Data Shape:{}".format(test_x.shape))
#%%
train_data = TensorDataset(torch.from_numpy(train_x) , torch.from_numpy(train_y))
val_data = TensorDataset(torch.from_numpy(val_x) , torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x) , torch.from_numpy(test_y))

batch_size=50
train_loader = DataLoader(train_data, shuffle = True , batch_size = batch_size)
val_loader = DataLoader(val_data, shuffle = True , batch_size = batch_size)
test_loader = DataLoader(test_data, shuffle = True , batch_size = batch_size)
#%%
# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
#%%

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
#%%
# Instantiate the model w/ hyperparams
vocab_size = len(word_to_int_dict)+1
output_size = 1
embedding_dim = 400 
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)
#%%
# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#%%
# training params

epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
       
        output, h = net(inputs.long(), h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in val_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs.long(), val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
#%%
# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    # get predicted outputs
    output, h = net(inputs, h)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
#%%
#Inference on a test review
# negative test review
test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'

from string import punctuation

def tokenize_review(test_review):
    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints

# test code and generate tokenized review
test_ints = tokenize_review(test_review_neg)
print(test_ints)


# test sequence padding
seq_length=200
features = pad_features(test_ints, seq_length)

print(features)

# test conversion to tensor and pass into your model
feature_tensor = torch.from_numpy(features)
print(feature_tensor.size())
#%%
def predict(net, test_review, sequence_length=200):
    
    net.eval()
    
    # tokenize review
    test_ints = tokenize_review(test_review)
    
    # pad tokenized sequence
    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)
    
    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)
    
    batch_size = feature_tensor.size(0)
    
    # initialize hidden state
    h = net.init_hidden(batch_size)
    
    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()
    
    # get the output from the model
    output, h = net(feature_tensor, h)
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze()) 
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    
    # print custom response
    if(pred.item()==1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")
#%%
# positive test review
# call function
seq_length=200 # good to use the length that was trained on

predict(net, test_review_neg, seq_length)