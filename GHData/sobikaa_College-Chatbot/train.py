import numpy as np
import json
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from chat import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

words_list = []
tags_list = []
pattern_list = []

for intent in intents['intents']:

    tag = intent['tag']
    tags_list.append(tag)

    for pattern in intent['patterns']:
        word = tokenize(pattern)
        words_list.extend(word)
        pattern_list.append((word, tag))

ignore_symbol = ['?', '.', '!', ',', '/', ':', ';', '-']

words_list = [stem(word) for word in words_list if word not in ignore_symbol]
words_list = sorted(set(words_list))
tags_list = sorted(set(tags_list))

x_train = []
y_train = []

for (pattern_sentence, tag) in pattern_list:
    bag = bag_of_words(pattern_sentence, words_list)
    x_train.append(bag)

    title = tags_list.index(tag)
    y_train.append(title)

x_train = np.array(x_train)
y_train = np.array(y_train)

num_epochs = 1000
batch_size = 15
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags_list)


class TrainingSet(Dataset):

    def __init__(self):
        self.n = len(x_train)
        self.x = x_train
        self.y = y_train

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n


dataset = TrainingSet()

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, titles) in train_loader:
        words = words.to(device)
        titles = titles.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, titles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {"model_state": model.state_dict(), "input_size": input_size, "hidden_size": hidden_size,
        "output_size": output_size, "words_list": words_list, "tags_list": tags_list}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
