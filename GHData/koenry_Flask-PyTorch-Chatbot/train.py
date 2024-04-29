import nltk, json, sqlite3, torch
nltk.download('punkt') # tokenizer package THIS SHOULD BE COMMENTED OUT AFTER DOWNLOADING!
from nltk.stem.porter import PorterStemmer
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from main import bagOfWords, tokenize, stem
from model import NeuralNetwork

with open('intents.json', 'r') as f:
    intents = json.load(f)

allWords = []
tags = []
xy = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        i = tokenize(pattern)
        allWords.extend(i) # extend of append so we dont put array of arrays
        xy.append((i, tag))
ignoreWords = ['?', '!', ',', '.'] # we should replace this with regex
# stemming
allWords = [stem(i) for i in allWords if i not in ignoreWords]
allWords = sorted(set(allWords)) # remove duplicates
tags = sorted(set(tags)) # sorting intent tags

xTrain = [] # words
yTrain = []  # tags
for (patternSentence, tag) in xy:
    bag = bagOfWords(patternSentence, allWords) # gets the tokenized sentence
    xTrain.append(bag)
    label = tags.index(tag)
    yTrain.append(label) # CAREFUL 1hot?
    # crossentropyloss ? so we dont have to care about 1hot encoding

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

class ChatDataset(Dataset):
    def __init__(self):
        self.nSamples = len(xTrain)
        self.xData = xTrain
        self.yData = yTrain

    #dataset[idx]
    def __getitem__(self, index):
        return self.xData[index], self.yData[index] # whats idx

    def __len__(self):
        return self.nSamples

# Hyperparameters
batch_size = 8
hiddenSize = 8
outputSize = len(tags)
inputSize = len(xTrain[0])
learningRate = 0.001
numEpochs = 1000

dataset = ChatDataset()
trainLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#Dataloader, automatically iterate over this and get training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(inputSize, hiddenSize, outputSize).to(device) # if gpu

# loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

# Train and Optimize
for epoch in range(numEpochs):
    for (words, labels) in trainLoader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item():.4f}')

data = {
    "modelState": model.state_dict(),
    "inputSize": inputSize,
    "outputSize": outputSize,
    "hiddenSize": hiddenSize,
    "allWords": allWords,
    "tags": tags,
}
# saving
FILE = "data.pth"
torch.save(data, FILE)

con = sqlite3.connect('ChatBot.db')
cur = con.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS ChatBot
    (DATE text, sentence text, correctResponse text)''')

consistent = input(f'training complete. file saved to {FILE} \n database created')









