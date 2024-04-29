import random
import json
import torch
from model import MyModel
from nltk_utils import tokenize, stem, bag_of_words
from data_collect import punc_words


# load the data
file_name = 'data.pth'
data = torch.load(file_name)

input_size = data['input_size']
model_state = data['model_state']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']

# load the data into mymodel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# load the intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Printing the chat entry statement
bot_name = 'Ahmed'
print("Let's start the chat, bro!")
print('If you wanna quit write "quit"')

while True:
    # Getting the input
    sentence = input('You: ')
    if sentence == 'quit':
        break

    # Applying modification to the sentence
    sentence = tokenize(sentence)
    sentence = [stem(w) for w in sentence if w not in punc_words]
    X = bag_of_words(sentence, all_words)

    # Reshaping so the model can accept it
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    # Predicting the values of tags and getting the max
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Checking the probabiltiy that it's the right tag
    probabilities = torch.softmax(output, dim=1)
    probability = probabilities[0][predicted.item()]

    # the bot will only response if the probability is higher than 0.7
    if probability.item() > 0.7:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
    else:
        print(f"{bot_name}: I don't understand dude!")
