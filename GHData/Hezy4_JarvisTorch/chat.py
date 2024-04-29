import random
import json
import torch
import os
import pyttsx3
from pyttsx3 import engine
import speech_recognition as sr
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import pyttsx3

engine = pyttsx3.init()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Jarvis"
print("Hello! Jarvis at your service! (type 'quit' to exit)")
engine = pyttsx3.init() # object creation

""" RATE"""
rate = engine.getProperty('rate')
engine.setProperty('rate', 200)

"""VOLUME"""
volume = engine.getProperty('volume')
engine.setProperty('volume',1.0)

"""VOICE"""
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
while True:
    with sr.Microphone() as source:
        r = sr.Recognizer()
        audio = r.listen(source)

            
        sentence = tokenize( r.recognize_google(audio))
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        r = sr.Recognizer()
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

            
        print("Recognizing:")
 
        try:
            print("You said:- " + r.recognize_google(audio))

            for intent in intents['cats']:
                if tag == intent["tags"]:
                    choice_words = random.choice(intent['responses'])
                    engine = pyttsx3.init()
                    engine.say(choice_words)
                    engine.runAndWait()
                    print(f"{bot_name}: {choice_words}")
    
        except sr.UnknownValueError:
            engine = pyttsx3.init()
            engine.say('I didnt get that. Rerun the code')
            print(f"{bot_name}: I do not understand...")
            engine.runAndWait()
