# %matplotlib inline
import os
import sys
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import music21
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
from music21 import instrument, note, stream, chord, tempo
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Generate a song starting from a given MIDI melody.')

parser.add_argument('--seed', type=str, default='72 74 76',
                    help='Initial notes as MIDI values, between 54 and 90, separated by spaces. 72 is C4. The input is cut off at the first 100 notes.')
parser.add_argument('--length', type=int, default='150', help='[1..n] Generated song length.')
parser.add_argument('--beta', type=float, default='15', help='[1..n] Softmax exponent (see report)')
parser.add_argument('--prob', type=float, default='1', help='[0..1] Probability to use softmax instead of argmax (see report)')
parser.add_argument('--plot', type=str, default='no', help='(yes/no) Some plots that show the frequency of the notes.')
parser.add_argument('--gpu', type=str, default='no', help='(yes/no) Load the model in GPU.')
parser.add_argument('--output_file', type=str, default='output', help='Named of the MIDI file given as output.')

args = parser.parse_args()

rng_seed = 42
random.seed(rng_seed)
np.random.seed(rng_seed)

if args.plot == 'yes':
    plot = True
else:
    plot = False

if args.gpu == 'yes':
    gpu = True
else:
    gpu = False


if args.seed == 'route29':
    midi_seed = [72, 74, 76, 76, 79, 79, 72, 74, 76, 72, 77, 76, 74, 71, 69, 67, 81, 79, 72, 74, 76, 76, 79, 79, 72, 74,
                 76, 72, 77, 76, 74, 74, 71, 72, 64, 64, 72, 74, 76, 76, 79, 79, 72, 74, 76, 72, 77, 76, 74, 71, 69, 67,
                 81, 79, 72, 74, 76, 76, 79, 79, 72, 74, 76, 72, 77, 76, 74, 74, 71, 72, 59, 60, 62, 69, 69, 72, 72, 77,
                 81, 79, 77, 67, 67, 72, 72, 76, 79, 77, 76, 65, 65, 69, 69, 74, 77, 76, 74, 76, 74, 73, 74, 73, 70, 69,
                 67, 69, 69, 72, 72, 77, 81, 79, 77, 67, 67, 72, 72, 76, 79, 77, 76, 77, 76, 74, 77, 76, 74, 59, 72, 67,
                 76, 67, 72, 67, 76, 67, 72, 67, 76, 67, 72, 72, 72, 74, 76, 76, 79, 79, 72, 74, 76, 72, 77, 76, 74, 71,
                 69, 67, 81, 79, 72, 74, 76, 76, 79, 79, 72, 74, 76, 72, 77, 76, 74, 74, 71, 72, 64, 64, 72, 74, 76, 76,
                 79, 79, 72, 74, 76, 72, 77, 76, 74, 71, 69, 67, 81, 79, 72, 74, 76, 76, 79, 79, 72, 74, 76, 72, 77, 76,
                 74, 74, 71, 72, 59, 60, 62, 69, 69, 72, 72, 77, 81, 79, 77, 67, 67, 72, 72, 76, 79, 77, 76, 65, 65, 69,
                 69, 74, 77, 76, 74, 76, 74, 73, 74, 73, 70, 69, 67, 69, 69, 72, 72, 77, 81, 79, 77, 67, 67, 72, 72, 76,
                 79, 77, 76, 77, 76, 74, 77, 76, 74, 59, 72, 67, 76, 67, 72, 67, 76, 67, 72, 67, 76, 67, 72, 72, 72, 74,
                 76, 76, 79, 79, 72, 74, 76, 72, 77, 76, 74, 71, 69, 67, 81, 79]
    # route 29

elif args.seed == 'route1':
    midi_seed = [72, 74, 76, 76, 76, 72, 74, 76, 76, 76, 72, 74, 76, 76, 77, 76, 74, 71, 72, 74, 74, 74, 71, 72, 74, 74,
                 74, 71, 72, 74, 74, 76, 74, 74, 76, 72, 72, 74, 76, 67, 76, 64, 76, 64, 72, 74, 76, 67, 76, 64, 76, 64,
                 72, 74, 76, 67, 76, 64, 77, 65, 65, 76, 74, 71, 72, 74, 77, 76, 74, 72, 71, 69, 71, 81, 74, 76, 76, 77,
                 79, 79, 76, 72, 84, 83, 81, 83, 79, 76, 72, 60, 76, 74, 59, 62, 59, 59, 76, 77, 79, 60, 79, 60, 76, 60,
                 72, 60, 84, 60, 83, 60, 81, 60, 65, 77, 79, 64, 84, 64, 83, 62, 86, 62, 84]
    # route 1

elif args.seed == 'route12':
    midi_seed = [67, 69, 67, 59, 59, 62, 63, 62, 59, 67, 69, 74, 78, 79, 74, 74, 86, 84, 83, 57, 57, 57, 81, 57, 81, 84,
                 83, 81, 83, 79, 74, 59, 59, 59, 59, 59, 59, 67, 63, 62, 59, 59, 59, 66, 59, 69, 59, 79, 74, 74, 86, 84,
                 83, 57, 57, 57, 81, 57, 84, 88, 86, 84, 86, 84, 83, 62, 62, 62, 62, 62, 62, 74, 72, 71, 79, 84, 67, 83,
                 69, 57, 81, 71, 57, 81, 74, 76, 77, 77, 72, 67, 69, 67, 72, 67, 81, 72, 83, 69, 81, 67, 79, 69, 79, 72,
                 74, 78, 79, 74, 66, 71, 59, 67, 69, 59, 71, 69, 79, 74, 81, 78, 57, 57, 74, 57, 57, 86, 57, 78, 76, 57,
                 76, 74, 62, 62, 72, 62, 62, 84, 62, 76, 78, 62, 83, 81, 79, 71, 59, 72, 59, 74, 59, 76, 78, 59, 74, 71,
                 69, 67, 69, 71, 72]
    # route12

elif args.seed == 'pallet':
    midi_seed = [79, 77, 76, 74, 84, 81, 83, 81, 79, 76, 72, 72, 74, 76, 77, 59, 71, 72, 74, 76, 77, 76, 74, 79, 77, 76,
                 79, 84, 83, 83, 84, 81, 79, 79, 60, 77, 76, 74, 72, 79, 77, 76, 74, 72, 72, 74, 76, 77, 60, 60, 79, 62,
                 59, 77, 76, 60, 60, 72, 74, 76, 77, 60, 77, 60, 79, 62, 59, 77, 79, 76, 60, 60, 62, 76, 74, 72, 74, 69,
                 76, 74, 72, 69, 71, 72, 76, 76, 74, 79, 77, 76, 74, 84, 81, 83, 81, 79, 76, 72, 72, 74, 76, 77, 59, 71,
                 72, 74, 76, 77, 76, 74, 79, 77, 76, 79, 84, 83, 83, 84, 81, 79, 79, 60, 77, 76, 74, 72, 79, 77, 76, 74,
                 72, 72, 74, 76, 77, 60, 60, 79, 62, 59, 77, 76, 60, 60, 72, 74, 76, 77, 60, 77, 60, 79, 62, 59, 77, 79,
                 76, 60, 60, 62, 76, 74, 72, 74, 69, 76, 74, 72, 69, 71, 72, 76, 76, 74]
    # pallet town

elif args.seed == 'melody1':
    midi_seed = [72, 74, 76, 77, 79, 81, 83, 84, 83, 81, 83, 86, 72, 74, 76, 77, 79, 81, 83, 84, 83, 81, 83, 86, 72, 74,
                 76, 77, 79, 81, 83, 84, 83, 81, 83, 86]

elif args.seed == 'melody2':
    midi_seed = [72, 72, 72, 72, 74, 74, 74, 74, 76, 76, 76, 76, 79, 79, 79, 79, 77, 77, 72, 72, 71, 72, 72, 72, 72, 72,
                 74, 74, 74, 74, 76, 76, 76, 76, 79, 79, 79, 79, 77, 77, 72, 72, 71, 72]

elif args.seed == 'melody3':
    midi_seed = [72, 74, 71, 72, 74, 74, 76, 71, 72, 72] * 10

elif args.seed == 'surf':
    midi_seed = [72, 71, 69, 67, 59, 57, 79, 79, 77, 74, 79, 67, 67, 76, 67, 67, 79, 77, 65, 65, 74, 65, 65, 77, 65, 65,
                 74, 81, 77, 79, 67, 67, 76, 67, 67, 79, 72, 72, 76, 71, 71, 79, 81, 69, 69, 77, 67, 67, 81, 71, 71, 79,
                 81, 79, 84, 74, 75, 84, 60, 60, 86, 84, 74, 77, 81, 71, 69, 79, 81, 71, 79, 76, 71, 69, 77, 76, 67, 74,
                 81, 69, 71, 79, 78, 69, 79, 84, 72, 74, 86, 84, 74, 77, 81, 71, 69, 79, 81, 71, 83, 84, 71, 72, 83, 81,
                 74, 79, 83, 76, 74, 84, 83, 72, 84, 88, 81, 79, 79, 67, 67, 76, 67, 67, 79, 77, 65, 65, 74, 65, 65, 77,
                 65, 65, 74, 81, 77, 79, 67, 67, 76, 67, 67, 79, 72, 72, 76, 71, 71, 79, 81, 69, 69, 77, 67, 67, 81, 71,
                 71, 79, 81, 79, 84, 74, 75, 84, 60, 60, 86, 84, 74, 77, 81, 71, 69, 79, 81, 71, 79, 76, 71, 69, 77, 76,
                 67, 74, 81, 69, 71, 79, 78, 69, 79, 84, 72, 74, 86, 84, 74, 77, 81, 71, 69, 79, 81, 71, 83, 84, 71, 72,
                 83, 81, 74, 79, 83, 76, 74, 84, 83, 72, 84, 88, 81, 79]
    # surf theme

elif args.seed == 'pokemon_center':
    midi_seed = [72, 67, 72, 79, 77, 69, 76, 74, 71, 65, 64, 67, 64, 65, 67, 71, 67, 71, 76, 74, 67, 71, 72, 76, 69, 71,
                 72, 71, 69, 67, 72, 67, 72, 79, 77, 69, 76, 74, 71, 65, 64, 67, 64, 65, 67, 71, 67, 71, 76, 74, 67, 71,
                 72, 62, 60, 62, 64, 65, 67, 69, 76, 62, 60, 55, 79, 64, 65, 67, 77, 79, 77, 76, 74, 64, 65, 67, 71, 64,
                 62, 65, 74, 60, 62, 65, 76, 77, 76, 74, 72, 55, 64, 55, 76, 71, 69, 55, 79, 69, 71, 72, 77, 76, 77, 79,
                 81, 71, 72, 74, 79, 69, 77, 76, 77, 67, 69, 65, 76, 77, 76, 74, 72, 62, 64, 65]
    # pokemon center

elif args.seed == 'show_me_around':
    midi_seed = [79, 77, 76, 77, 76, 74, 72, 71, 67, 67, 71, 74, 77, 79, 72, 76, 79, 77, 81, 84, 72, 76, 79, 77, 76, 74,
                 72, 76, 79, 77, 81, 84, 84, 83, 79, 81, 83, 84, 84, 84, 88, 84, 84, 79, 81, 84, 84, 79, 84, 84, 86, 76,
                 83, 74, 83, 71, 88, 84, 84, 79, 81, 84, 84, 79, 89, 86, 84, 83, 84, 84, 84, 88, 84, 84, 79, 81, 84, 84,
                 79, 84, 84, 86, 76, 83, 74, 83, 71, 88, 84, 84, 79, 81, 84, 84, 79, 89, 86, 84, 83, 84, 84, 84, 88, 84,
                 84, 79, 81, 84, 84, 79, 84, 84, 86, 76, 83, 74, 83, 71, 88, 84, 84, 79, 81, 84, 84, 79, 89, 86, 84, 83,
                 84, 84, 84, 88, 84, 84, 79, 81, 84, 84, 79, 84, 84, 86, 76, 83, 74, 83, 71, 88, 84, 84, 79, 81, 84, 84,
                 79, 89, 86, 84, 83, 84, 84, 84]
    # show me

elif args.seed == 'major_scale':
    midi_seed = [72, 74, 76, 77, 79, 81, 83, 84] * 13

elif args.seed == 'penta_scale':
    midi_seed = [72, 74, 76, 81, 83, 84] * (int(24 / 6) + 1)

else:
    midi_seed = list(map(lambda x: int(x), args.seed.split(" ")))

midi_length = args.length

QUANT = 1 / 8  # 0.0625
LOWEST_NOTE = 60 - 6
HIGHEST_NOTE = 60 + 36 - 6

layers_num = 3
dropout_prob = 0.3
hidden_units = 1024
input_size = HIGHEST_NOTE - LOWEST_NOTE + 1
alphabet_len = input_size


def freq_analysis(song, lbl):
    notes_count = [0] * 12
    for n in song:
        notes_count[n % 12] += 1
    plt.figure(figsize=(5, 1))
    plt.bar(["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"], notes_count, label=lbl)
    plt.legend(loc='upper right')


def create_midi(prediction_output, name=''):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    mm1 = tempo.MetronomeMark(number=180)
    output_notes.append(mm1)
    # create note and chord objects based on the values generated by the model
    for n in prediction_output:
        new_note = note.Note(int(n))
        new_note.offset = offset
        new_note.duration.quarterLength = QUANT
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5  # QUANT

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=name + '.mid')


class Network(nn.Module):
    def __init__(self, input_size, hidden_units, layers_num, seq_size=100, dropout_prob=0.3):
        # Call the parent init function (required!)
        super().__init__()
        # Define recurrent layer
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True
                           )
        # Define output layer
        self.lin1 = nn.Linear(hidden_units * 2, 256 * 2)
        self.lin2 = nn.Linear(256 * 2, input_size)
        self.lin = nn.Linear(hidden_units, input_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.batchNorm1 = nn.BatchNorm1d(num_features=hidden_units)
        self.batchNorm2 = nn.BatchNorm1d(256 * 2)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, state=None):
        # LSTM
        x, rnn_state = self.rnn(x, state)  # (batch_size, seq_len, hidden_size).

        # for some reason i can't do batch normalization with the current axes.
        # batch normalization helps with gradient vanishing.
        x = x.permute(0, 2, 1)
        x = self.batchNorm1(x)
        x = x.permute(0, 2, 1)
        # helps to generalize
        x = self.dropout(x)

        # Linear layer
        x = self.lin(x)  # 1(x)

        # use logsoftmax to compute logits.
        x = self.softmax(x)
        return x, rnn_state


def create_one_hot_matrix(encoded, alphabet_len):
    tot_chars = len(encoded)

    try:
        # Create one hot matrix
        encoded_onehot = np.zeros([tot_chars, alphabet_len])

        encoded_onehot[np.arange(tot_chars), encoded] = 1
    except:
        print(encoded)
        raise

    return encoded_onehot


def create_one_hot_matrix_tensor(encoded_tensor, alphabet_len):
    tot_chars = len(encoded_tensor)
    encoded_onehot = torch.zeros([tot_chars, alphabet_len])
    encoded_onehot[torch.arange(tot_chars), encoded_tensor] = 1
    return encoded_onehot


class OneHotEncoder():

    def __init__(self, alphabet_len):
        self.alphabet_len = alphabet_len

    def __call__(self, sample):
        # Load encoded text with numbers
        encoded = np.array(sample['encoded'])

        # Create one hot matrix
        encoded_onehot = create_one_hot_matrix(encoded, self.alphabet_len)
        return {**sample,
                'encoded_onehot': encoded_onehot}


class ToTensor():
    def __call__(self, sample):
        # Convert one hot encoded text to pytorch tensor
        encoded_onehot = torch.tensor(sample['encoded_onehot']).float()
        return {'encoded_onehot': encoded_onehot}


def encode_song(song):
    result = []
    for s in song:
        result.append(s - LOWEST_NOTE)  # encode_note(s))
    return result


def encode(x):
    return x - LOWEST_NOTE


def decode(x):
    return x + LOWEST_NOTE


def load_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)

    if not gpu:
        checkpoint = torch.load("net_params.pth", map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load("net_params.pth")

    net = Network(input_size, hidden_units, layers_num, dropout_prob)
    net.to(device)
    net.load_state_dict(checkpoint)
    return net


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Selected device:', device)

net = load_model()


def decode(tensor):
    return tensor.item() + LOWEST_NOTE  # decode_note(tensor.item())


net = net.eval()
generated_song = []
output_song = []

# %% Find initial state of the RNN
with torch.no_grad():
    # Encode seed
    seed_encoded = midi_seed

    if len(seed_encoded) > 100:
        seed_encoded = seed_encoded[:99]

    seed_encoded = encode_song(seed_encoded)

    # One hot matrix
    seed_onehot = create_one_hot_matrix(seed_encoded, alphabet_len)
    # To tensor
    seed_onehot = torch.tensor(seed_onehot).float()
    # Add batch axis
    seed_onehot = seed_onehot.unsqueeze(0).to(device)

    for n in seed_encoded:
        generated_song.append(n + LOWEST_NOTE)  # decode_note(n))

    seed_song = list(generated_song)
    print("SEED: ", generated_song)
    # Forward pass
    net_out, net_state = net(seed_onehot)
    # Get the most probable last output index
    next_char_encoded = net_out[:, -1, :].argmax().item()
    generated_song.append(next_char_encoded + LOWEST_NOTE)
    output_song.append(next_char_encoded + LOWEST_NOTE)

# %% Generate sonnet
new_line_count = 0
tot_char_count = 0
p_sample = args.prob  #1
exponent = args.beta  #14
while True:
    with torch.no_grad():  # No need to track the gradients
        # The new network input is the one hot encoding of the last chosen letter

        net_input = create_one_hot_matrix_tensor([next_char_encoded], alphabet_len)
        net_input = net_input.clone().detach().float()
        net_input = net_input.unsqueeze(0).to(device)
        # Forward pass
        net_out, net_state = net(net_input, net_state)
        # Get the most probable letter index

        output_softmax = torch.nn.Softmax(dim=-1)(torch.exp(net_out[0][0]))
        prob_transformed = (output_softmax ** exponent) / torch.sum(output_softmax ** exponent)

        prob_dist = torch.distributions.Categorical(prob_transformed)  # probs should be of size batch x classes

        sampled_value = prob_dist.sample()

        if random.random() < p_sample:
            next_char_encoded = sampled_value
        else:
            next_char_encoded = net_out[:, -1, :].argmax()

        # Decode the note
        next_char = decode(next_char_encoded)

        generated_song.append(next_char)
        output_song.append(next_char)
        # Count total notes
        tot_char_count += 1

        if tot_char_count > midi_length:
            print()
            print("Generated song: \n", generated_song)
            break

create_midi(generated_song, name=args.output_file)
print("=" * 60)
print("File written to disk! ({0}.mid)".format(args.output_file))

freq_analysis(seed_song, "Seed")
freq_analysis(generated_song, "Generated")

if plot:
    plt.show()
