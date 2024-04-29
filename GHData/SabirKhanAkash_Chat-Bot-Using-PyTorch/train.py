import json
from NLTK_init import tokenize, stem, BagOfWords
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r' ) as f:
	intents = json.load(f)

allWords = []
tags = []
xy = []
for intent in intents['intents']:
	tag = intent['tag']
	tags.append(tag)
	for pattern in intent['patterns']:
		w = tokenize(pattern)
		allWords.extend(w)
		xy.append((w,tag))

ignoreWords = ['?', '!', '.', ',']
allWords = [stem(w) for w in allWords if w not in ignoreWords]
allWords = sorted(set(allWords))
tags = sorted(set(tags))


xTrain = []
yTrain = []

for (patternSentence, tag) in xy:
	bag = BagOfWords(patternSentence, allWords)
	xTrain.append(bag)

	label = tags.index(tag)

	yTrain.append(label)

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

class ChatDataSet(Dataset):
	def __init__(self):
		self.n_samples = len(xTrain)
		self.x_data = xTrain
		self.y_data = yTrain

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]
		
	def __len__(self):
		return self.n_samples

batchSize = 8
hiddenSize = 6
outputSize = len(tags)
inputSize = len(xTrain[0])
learningRate = 0.1
numEpochs = 1000

print(inputSize, len(allWords))
print(outputSize, tags)


dataset = ChatDataSet()
trainLoader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(inputSize, hiddenSize, outputSize)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

for epoch in range(numEpochs):
	for (words, labels) in trainLoader:
		words = words
		labels = labels

		outputs = model(words)
		loss = criterion(outputs, labels.long())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if (epoch +1) % 10 == 0:
		print(f'epoch {epoch+1}/{numEpochs}, loss = {loss.item():.3f}')

print(f'final loss, loss = {loss.item():.3f}')

data = {
	"modelState": model.state_dict(),
	"inputSize": inputSize,
	"outputSize": outputSize,
	"hiddenSize": hiddenSize,
	"allWords": allWords,
	"tags":tags
}

FILE = "data.pth"
torch.save(data,FILE)

print(f'training complete, file has been saved to {FILE}')
