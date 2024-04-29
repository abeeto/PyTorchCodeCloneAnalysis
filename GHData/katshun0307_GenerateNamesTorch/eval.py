# -*- coding: utf-8 -*- #

""" make predictions using the network
"""

import torch
import torch.nn

import os

import dataset
from network import RNN

dataset = dataset.Dataset()

max_length = 20

rnn = RNN(dataset.n_letters, 128, dataset.n_letters, dataset.n_categories)
rnn.eval()

# load weights
if os.path.exists("models/gen_names.pkl"):
    checkpoint = torch.load("models/gen_names.pkl")
    rnn.load_state_dict(checkpoint['nn_state_dict'])
    print("checkpoint loaded")


def sample(category, start_char):
    with torch.no_grad():
        category_tensor_var = dataset.category_tensor(category)
        input = dataset.input_tensor(start_char)
        hidden = rnn.init_hidden()
        output_name = start_char

        for i in range(max_length):
            output, hidden = rnn(category_tensor_var, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == dataset.n_letters - 1:
                break
            else:
                letter = dataset.all_letters[topi]
                output_name += letter
            input = dataset.input_tensor(letter)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


samples("English", "ENG")
samples("Russian", "RUS")
