# 1. Theory + NLP  concepts (stemming, tokenization, bag of words)
# 2. Create training data
# 3. PyTorch model and training
# Save/load model and implement the chat

# def run():
#     torch.multiprocessing.freeze_support()
#     print('loop')

# if __name__ == '__main__':
    #run() all code here if it gives runtime error of freezesupport() 



import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

# pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader  # look for pytorch beginner course Dataset and Dataloaders

# import model
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)
# print(intents)

all_words = []  # To collect all 
tags = [] # To collect the tags
xy = [] # To hold both patterns and tags 

for intent in intents['intents']:
    tag = intent['tag']  #get the tag from json file
    tags.append(tag)   # append in tags array

    for pattern in intent['patterns']:  # loop over different patterns
        w = tokenize(pattern)   # apply tokenization from utility function
        all_words.extend(w)  # put into all_words array, extending because w is also 
        xy.append((w, tag))  # put in tokenized pattern and corresponding label to the xy list (a tuple with pattern and corresponding tag) 

# punctuation characters we want to ignore 
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]  # apply stemming. Stem each word for w in all words and also exclude the ignore words by list comprehension, w not in ignore words. All words in lower cases and ending chopped
# print(all_words)
all_words = sorted(set(all_words))  # sort these words and we also only want unique words by convert to set..will return list
tags = sorted(set(tags))  # also sort tags..will have unique labels
# print(tags)


# To create the training data, continuing in our pipeline to create bag of words
# Create the list with x_train data and y_train data

X_train = [] # bag of words
y_train = [] # tags, numbers  # for tags or associated numbers for tag

# Loop over the xy array, and unpack the tuple with pattern and tag
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)  # create bag of words
    X_train.append(bag)  # append to training data

    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)  # For the Y data, it will the labels, it is tags. First tag will give label 0, next 1. We have numbers for our labels
    y_train.append(label) # we don't need to take care of 1 hot encoding here, because later on we implement CrossEntropyLoss


# Convert into numpy arrays
# We now have the training data
X_train = np.array(X_train)
y_train = np.array(y_train)


# We have to Implement the bag_of_words inside utility

# create class which implements _init_(self)
class ChatDataset(Dataset):  # inherit Dataset
    def __init__(self):
        self.n_samples = len(X_train)  # number of samples
        self.x_data = X_train   # X training array
        self.y_data = y_train

    # dataset[indx] # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # length method
    def __len__(self):
        return self.n_samples

# hyperparameters
batch_size = 8
input_size = len(X_train[0])  # First bag of words...length of bag of words
hidden_size = 8
output_size = len(tags)  # number of different classes or tags we have
learning_rate = 0.001
num_epochs = 1000
# print(input_size, len(all_words))
# print(output_size, tags)

dataset = ChatDataset()  #dataset
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Create dataloader, num_workers for multiprocesing # num_workers 2 if in mac, gives error in windows So use 0 for windows

# check for gpu support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)  # labels to scalar type Long


        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)  # labels Expected to be object of scalar type Long

        # backward and optimizer step
        optimizer.zero_grad()  # empty the gradients for pytorch optimization
        loss.backward()  # calculate backpropagation
        optimizer.step()

    # print current epoch and loss in every 100 steps
    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')

print(f'final loss, loss = {loss.item():.4f}')


# create a dictionary and save in it
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"  # pth for pytorch
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')