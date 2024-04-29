import random
import json
import torch 
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# check for gpu support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# open the json file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# open our saved file
FILE = "data.pth"
data = torch.load(FILE)  # load our file

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)  # load the model_state_dict
model.eval()                        # set evaluation mode

bot_name = 'Wulfi'
print("Let's chat! type 'quit' to exit ")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])    # 1 row for one sample, shape[0] as column
    X = torch.from_numpy(X).to(device)  # bag_of_words function returns numpy array

    output = model(X)   # gives the prediction
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]   # get the actual tag from predicted.item()

    # to get the actual probability of tag
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]   # probability


    if prob.item() > 0.75:  
        # find the corresponding intents... loop over all the intents and check if tag matches
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f'{bot_name}: I didn\'t get you...')