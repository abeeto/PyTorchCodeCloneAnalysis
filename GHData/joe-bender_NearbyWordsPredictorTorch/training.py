import torch
import torch.nn as nn
import torch.optim as optim
import helpers
from hyperparameters import hps
import random

class NearbyWords(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(NearbyWords, self).__init__()
        self.hidden = nn.Linear(vocab_size, embed_size, bias=False)
        self.output = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output

words = helpers.get_words(hps['filename'])
vocab = set(words)
word_to_int, int_to_word = helpers.get_word_mappings(words)
pairs = helpers.get_training_pairs(words, 3)

vocab_size = len(vocab)
embed_size = hps['embed_size']

model = NearbyWords(vocab_size, embed_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=hps['learning_rate'])

for t in range(hps['epochs']):
    pair_sample = random.sample(pairs, 1000)

    losses = []
    for x, y in pair_sample:
        x = helpers.index_to_1hot(word_to_int[x], vocab_size).view(1, -1)
        y = torch.tensor([word_to_int[y]])

        y_pred = model(x)
        loss = criterion(y_pred, y)
        losses.append(loss)
    losses = torch.stack(losses)
    loss = torch.mean(losses)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print loss at intervals
    if (t+1) % hps['print_every'] == 0 or t == 0:
        print(t+1, loss.item())
