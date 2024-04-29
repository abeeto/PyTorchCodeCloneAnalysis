import numpy as np
import random
import json
import torch
import nltk
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt  

import string
import warnings
import sys
warnings.filterwarnings("ignore")

try :
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try :
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
with open('intents.json', 'r') as f:
    intents = json.load(f)
    #print(intents)
all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))


# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)



# Hyper-parameters 
num_epochs = 2000
batch_size = 3
learning_rate = 0.0001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)
print("xtranin\n",X_train)
print("ytrain\n",y_train)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True, num_workers=2)
loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

print( "the train loader\n", dataset)                    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = NeuralNet(input_size, hidden_size, output_size).to(device)
print("show the model\n", model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print("the optimazer is as folws\n",optimizer)
def acc_cal(loader,model):
	num_corrects=0
	num_samples=0
	model.eval()
	with torch.no_grad():
		for words,labels in train_loader:
			words=words.to(device)
			labels=labels.to(dtype=torch.long).to(device)
			#prepare the data for the model
			#words=words.reshape(-1,510)
			#forward
			outputs=model(words)
			#acuracy calculation
			_,predictions=outputs.max(1)
			num_corrects += (predictions==labels).sum()
			num_samples +=predictions.size(0)
		print(f"Accuracy ={num_corrects/num_samples*100:.4f}: Received {num_corrects}/{num_samples}")
		model.train()
# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        #labels = torch.max(labels, 1)[0]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                   
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        acc_cal(train_loader,model)
        
       
print(f'final loss: {loss.item():.4f}')
#plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')  
#plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')  
#plt.title('Google Stock Price Prediction')  
 
#plt.show()  
#plt.plot(input_size,label='Running Loss History')  
#plt.plot(hidden_size,label='Running correct History') 
#plt.plot(output_size,label='Running correct History') 

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}
#plt.title('loop through each sentence in our intents patterns') 
#plt.plot(w,label='stemed words') 
#plt.plot(tags,label='tags') 
#plt.xlabel('stemed words')  
#plt.ylabel('tags')  
#plt.legend() 
#plt.show()
#print("model_state\n",model.state_dict())
#print("input_size\n",input_size,)
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
