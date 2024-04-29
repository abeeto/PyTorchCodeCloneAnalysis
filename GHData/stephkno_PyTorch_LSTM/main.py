#!/usr/local/anaconda3/envs/experiments/bin/python3

import torch
import sys
import datetime
from termcolor import colored
import atexit
import keyboard
import random

#model
class model(torch.nn.Module):
    def __init__(self, hidden_size, rate):
        super(model, self).__init__()

        self.rnn = torch.nn.LSTM(alphabet_size, hidden_size, n_layers, batch_first=True)
        self.decoder = torch.nn.Linear(hidden_size, alphabet_size)

        self.loss = torch.nn.BCELoss()
        print("Learning rate: ", rate)
        self.optim = torch.optim.Adam(lr=float(rate), params=self.parameters())
        self.reset_hidden()

    def reset_hidden(self):
        self.h = (torch.zeros(n_layers, 1, hidden_size).cuda(),
                  torch.zeros(n_layers, 1, hidden_size).cuda())

    def forward(self, input_vec):
        input_vec = input_vec.detach().unsqueeze(0)
        context = self.h[0].detach(), self.h[1].detach()

        x, context = self.rnn(input_vec, context)

        self.h = context
        h_ = torch.nn.Dropout(0.0)(x)
        d = self.decoder(h_)
        d = torch.nn.Softmax(dim=2)(d/temperature)
        self.reset_hidden()
        return h_, d

#load dataset
filename = sys.argv[1]
text = []
alphabet = []

#open file
with open(filename, "r") as f:
    # reads all lines and removes non alphabet words
    book = f.read()

f.close()

book = list(book)
text = []

#parse book
for t in book:
    if t == "\n":
        t = "¶"
    #print(t,end="")
    text.append(t)

for i,e in enumerate(book):
    if e.lower() not in alphabet:
        alphabet.append(e.lower())

del book

#sort/format tokens
alphabet.sort()
alphabet[alphabet.index("\n")] = "¶"
alphabet_size = len(alphabet)

epochs = 0

#parameters
hidden_size = 512
n_chars = 1
n_layers = 2
sequence_length = 1000
step = 0
steps = 10000
rate = 0.0025
c = 0
total_loss = 0.0
n_correct = 0
temperature = 1.0
epsilon = 1.0

render = False
show_grad = False
teacher_force = True
first = True

one_hot_vecs = {}
p_avg = 0
avg_loss = 0

for i in alphabet:
    t = torch.zeros(alphabet_size).cuda()
    t[alphabet.index(i.lower())] = 1.0
    one_hot_vecs[i] = t

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('GRU') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def one_hot(char):
    return one_hot_vecs[char]

def get_next_seq(character):
    c = character
    new = False

    if c >= len(text):
        c = 0
        new = True

    out = []
    target = []

    for i in range(sequence_length):
        char = one_hot(text[(c+i)%len(text)].lower())
        out.append(char)
    for i in range(sequence_length):
        char = one_hot(text[(c+i+1)%len(text)].lower())
        target.append(char)
    """
    print("\n")
    for o in out:
        print(alphabet[torch.argmax(o)],end="")
    print("\n")
    for o in target:
        print(alphabet[torch.argmax(o)],end="")
    print("\n-\n")
    """
    return out, torch.stack(target).cuda(), c+sequence_length+1, new


def savemodel():
    print("Save model parameters? [y/n]➡")
    filename_input = input()

    if filename_input == 'y' or filename_input == 'Y' or filename_input.lower() == 'yes':
        filename = "Model-" + str(datetime.datetime.now()).replace(" ", "_")
        print("Save as filename [default: {}]➡".format(filename))

        filename_input = input()
        if not filename_input == "":
            filename = "Model-" + str(filename_input).replace(" ", "_")

        print("Saving model as {}...".format(filename))
        modelname = "./models/{}".format(filename)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optim.state_dict()
        }, modelname)

    # print("Best parameters:")
    # print(best)
    quit()

atexit.register(savemodel)

#init model
model = model(hidden_size, rate).cuda()
weights_init(model)

counter = 1

start = step
out_text = []

c = 1

for a in alphabet:
    print(a,end=" ")

print(" - Size:{}\n".format(alphabet_size))

#train loop
while True:
    if False:
        for p in model.parameters():
            print(p)
            print(p.grad)
    elif keyboard.is_pressed(' '):
        teacher_force = not teacher_force
        if teacher_force:
            print("\nForce on")
        else:
            print("\nForce off")
        print("")

    inp, target, c, new = get_next_seq(c)

    a = 0

    if new:
        model.reset_hidden()
        if epochs % 1 == 0 and not render:
            print("")
            for i in range(int(len(out_text))):
                print(out_text[i],end="")
            print("")

        out_text.clear()
        avg_loss = round(total_loss / len(text), 4)
        accuracy = int(100 * (n_correct / counter))

        out = colored(
            "[Epoch{}|Progress:[{}%]|[{}]|loss:{}|avg:{}|{}|Acc:{}%|Eps:{}]".format(epochs, progress, a, total_loss,
                                                                                    avg_loss, indicator, n_correct, epsilon), attrs=['reverse'])
        n_correct = 0
        counter = 0
        total_loss = 0
        epochs += 1
        first = True

    inp = torch.stack(inp)

    d, out = model.forward(inp)

    probs = torch.distributions.Categorical(out.squeeze(0))
    sample = probs.sample().cuda()

    outchar = [alphabet[s] for s in sample]
    targetchar = [alphabet[torch.argmax(t)] for t in target]

    if not teacher_force:
        next_inp = []
        for o in outchar:
            next_inp.append(one_hot(o))

    done = False

    loss = model.loss(out, target.unsqueeze(0))
    loss.backward(retain_graph=True)

    total_loss += loss.item()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    model.optim.step()
    model.optim.zero_grad()

    counter += 1

    p_avg = avg_loss
    if avg_loss > p_avg:
        indicator = "⬆"
    else:
        indicator = "⬇"

    for o in outchar:
        out_text.append(o)

    progress = round(100*(c/len(text)),1)
    if not render and step % steps == 0:
        print("\r", progress, end="%")

    if targetchar == "¶":
        sys.stdout.flush()

    if render:
        #if counter % 10 == 0:
        #    out = colored("\r[Epoch{}|Progress:[{}%]|[{}]|loss:{}|avg:{}|{}|Acc:{}%]".format(epochs, progress, a, 0, avg_loss, indicator, accuracy),attrs=['reverse'])
        #    print(out,end="")
        #    sys.stdout.flush()

        for i in outchar:
            if i == "¶":
                i = "\n"
            print(i, end="")
            sys.stdout.flush()
    else:
        print("\r",progress,end="%")
    torch.cuda.empty_cache()
