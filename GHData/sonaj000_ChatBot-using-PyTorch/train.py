import json 
from nltk_util import tokenize,stemming,bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from model import NeuralNet

with open('test.json','r') as f: #json file needs to be out of the same folder as the python scripts
    intents = json.load(f)

trigger_words = []
tags = []
xy = []
ignore_words = ['?','!','.',',']

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        trigger_words.extend(w)
        xy.append((w,tag))

trigger_words = [stemming(w) for w in trigger_words if w not in ignore_words]
trigger_words = sorted(set(trigger_words)) #returns list of unique words
tags = sorted(set(tags))


X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence,trigger_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label) #crossentropy loss

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

#Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#TRaining the model
for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        #forward pass
        outputs= model(words)

        loss = criterion(outputs,labels)

        #backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"trigger_words": trigger_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')