import random, torch, json, re
from model import NeuralNetwork
from main import bagOfWords, tokenize

def chat(sentence):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use gpu if possible
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

    model = NeuralNetwork(inputSize, hiddenSize, outputSize).to(device) # if gpu
    model.load_state_dict(modelState)
    model.eval()
    # load and evaluate model

    botName = 'Big Bad Bot'
    print('Lets Talk! type quit to eit')
    while True:
        
        sentence =  tokenize(re.sub(r'[0-9]+', '', (sentence) ))
        X = bagOfWords(sentence, allWords)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device) # because returns a numpy array

        output = model(X)
        _, predicted =  torch.max(output, dim=1) # why _,
        tag = tags[predicted.item()]

        # softmax probabiltiy
        probs = torch.softmax(output, dim=1)
        chat.prob =  probs[0][predicted.item()]

        if chat.prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    chat.answerBot = f'{botName}: {random.choice(intent["responses"])}'
                   # Everything in python is considered as object so functions are also objects. So you can use this method as well.
                    return json.dumps(chat.prob.item())
        else: 
            print(f'{botName}: I do not understand')
            return json.dumps(chat.prob.item())
