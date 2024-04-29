import pandas as pd  
import random
import torch 
from model import NeuralNet, device
from nltk_utils import bag_of_words, tokenize 

BOT_NAME = "Joi"
FILE = "brain.pth" 


model_state = torch.load(FILE)
chat_data = pd.read_csv("database.csv")
response_data = {tag.strip():sentence for tag, sentence in zip(chat_data["tag"].values, chat_data["sentence"].values)}

input_size = model_state["input_size"]
hidden_size = model_state["hidden_size"]
output_size = model_state["output_size"]
all_words = model_state["all_words"]
tags = model_state["tags"]
model_state = model_state["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)

model.load_state_dict(model_state)

while True:
    sentence = input("Humano:")
    X = bag_of_words(tokenize(sentence), all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    output = model(X)
    
    # Se obtiene el valor maximo de salida
    value, predicted = torch.max(output, dim=1)
     
    # Se calcula cual eficaz es el maximo valor
    probs = torch.softmax(output, dim=1)
    tag = tags[predicted.item()]
    prob = probs[0][predicted.item()]
    print(f'probablemente sea un {tag} con %{prob}')

    tag = tag.replace("pregunta","respuesta")
    tag = tag.replace("afirmacion", "respuesta") 

    if prob.item() > 0.75:
        print(f'{BOT_NAME}:{response_data[tag]}')
        pass
    else:
        print(f"{BOT_NAME}: I dont understand.")
    
    


