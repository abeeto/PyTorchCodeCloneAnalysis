import random
import json
import torch
from model import NeuralNet
from NLTK_init import BagOfWords, tokenize
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
	intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

inputSize = data["inputSize"]
hiddenSize = data["hiddenSize"]
outputSize = data["outputSize"]
allWords = data["allWords"]
tags = data["tags"]
modelState = data["modelState"]

model = NeuralNet(inputSize, hiddenSize, outputSize)
model.load_state_dict(modelState)
model.eval()

bot_name = "Bingo"
print("\nWelcome to Bingo's Chitchat! Let's chat! Type 'quit' to exit")

while 1==1:
	sentence = input('You: ')
	if sentence == 'quit':
		break

	sentence = tokenize(sentence)
	X = BagOfWords(sentence, allWords)
	X = X.reshape(1,X.shape[0])
	X = torch.from_numpy(X)

	output = model(X)
	_, predicted = torch.max(output, dim=1)
	tag = tags[predicted.item()]

	probs = torch.softmax(output, dim=1)
	prob = probs[0][predicted.item()]

	if prob.item() > 0.75:
		for intent in intents["intents"]:
			if tag == intent["tag"]:
				print(f"{bot_name}: {random.choice(intent['responses'])}")
	else:
		print(f"{bot_name}: I do not understand . . .")