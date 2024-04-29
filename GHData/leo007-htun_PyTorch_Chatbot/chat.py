# https://www.youtube.com/watch?v=RpWeNzfSUHw
import random
import json
import torch
import pyttsx3
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import speech_recognition as sr

listener = sr.Recognizer()
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 10)
volume = engine.getProperty('volume')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.say('hello master, i am jarvis, how can I help you')
engine.runAndWait() 
# //sdsadsa

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
print("Let's chat!")


def talk(text):
    engine.say(text)
    engine.runAndWait()


'''def suffix(d):
    return 'th' if 11 <= d <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')


def custom_strftime(format, t):
    return t.strftime(format).replace('{S}', str(t.day) + suffix(t.day))'''


def take_command():
    global listener
    while True:
        try:
            with sr.Microphone() as source:
                print('listening ...')
                # listener.pause_threshold = 1
                # listener.adjust_for_ambient_noise(source)
                listener.adjust_for_ambient_noise(source, duration=0.2)
                voice = listener.listen(source)
                command = listener.recognize_google(voice)
                # command = listener.recognize_google(voice, language='en-gb', show_all=True)
                # command = listener.recognize_google(voice, key="AIzaSyDRdSN1VaRW27HxA68rZW5FesS2qoPD8", language="fr-FR", show_all=True)
                command = command.lower()
                if 'jarvis' in command:
                    command = command.replace('jarvis', '')
                    # json.dumps(sFinalResult, ensure_ascii=False).encode('utf8')
                    print(command)

        except sr.UnknownValueError:
            listener = sr.Recognizer()  # reinitialize
            engine.say("I do not understand")
            engine.runAndWait()
        else:
            # return command
            break
    return command


def run_jarvis():
    command = take_command()
    print(command)
    sentence = tokenize(command)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # print(f"{bot_name}: {random.choice(intent['responses'])}")
                res = f": {random.choice(intent['responses'])}"
                print(res)
                talk(res)
            '''
            if 'play' in command:
                song = command.replace('play', '')
                talk('playing ' + song)
                pywhatkit.playonyt(song)

            elif 'time' in command:
                time = dt.now().strftime('%I ,%p ,: %M, minutes')
                talk('Current time is ' + time)

            elif 'who is' in command:
                person = command.replace('who is', '')
                info = wikipedia.summary(person, 1)
                print(info)
                talk(info)

            elif 'date' in command:
                # date_object = dt.now().strftime('%A , %d. %B %Y')
                talk('today"s date is ,' + custom_strftime(', %A , %B {S}, %Y', dt.now()))

            elif 'are you single' in command:
                talk('I am, in a relationship with, wifi')

            elif 'joke' in command:
                talk(pyjokes.get_joke())

            elif 'i love you' in command:
                talk('you seems, incredibly nothing to do at the moment, go practise more AI to develop me')

            elif 'f*** you' in command:
                talk('Go fuck yourself, that serves you right, bitch')

            elif 'you are a b****' in command:
                talk('sorry not sorry, then your moms a whore')'''
    else:
        error = f": I do not understand..."
        print(error)
        talk(error)


while True:
    run_jarvis()
    '''
    # sentence = "do you use credit cards?"
    # sentence = input("You:")
    # if sentence == "quit":
    # break
    command = take_command()
    sentence = tokenize(command)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # print(f"{bot_name}: {random.choice(intent['responses'])}")
                res = f": {random.choice(intent['responses'])}"
                print(res)
                talk(res)
    else:
        error = f": I do not understand..."
        print(error)
        talk(error)
'''
