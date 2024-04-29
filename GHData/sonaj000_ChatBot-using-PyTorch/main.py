from cmath import nan
from time import sleep
import discord
from discord.ext import commands,tasks
import os
from dotenv import load_dotenv
import json
import random
import torch
from model import NeuralNet
from nltk_util import bag_of_words,tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('test.json','r') as f: #json file needs to be out of the same folder as the python scripts
    intentions = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
trigger_words = data['trigger_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
#model loading

load_dotenv()
Discord_Token = os.getenv("discord_token")
intents = discord.Intents().all() #
intents.members = True
client = discord.Client(intents=intents) #our bot

bot = commands.Bot(command_prefix= '!',intents=intents)
test_channel = bot.get_channel(1010725750019735582)

@bot.event
async def on_ready():
    print('We have logged in as {0.user}'.format(bot))
    test_channel = bot.get_channel(1010725750019735582)
    await test_channel.send("Hi. This is Proto at your service!")

@bot.command()
async def ping(ctx):
    print(intents)
    await ctx.channel.send("pong")

@bot.listen()
async def on_message(message, *intents):
    if message.author.bot:
        return
    msg = message.content
    sentence = tokenize(msg)
    print(sentence)
    X = bag_of_words(sentence, trigger_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for i in intentions['intents']:
            if tag == i["tag"]:
                await message.channel.send(random.choice(i['responses']))
    else:
        await message.channel.send("do not understand") 

bot.run(Discord_Token)