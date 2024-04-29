import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

# hyperparameter
K = 100  # number of negative samples
C = 5  # nearby words threshold
NUM_EPOCHS = 30  # The number of epochs of training
MAX_VOCAB_SIZE = 50000  # the vocabulary size
BATCH_SIZE = 128  # the batch size
LEARNING_RATE = 0.001  # the initial learning rate
EMBEDDING_SIZE = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenize function, convert a text into words
def word_tokenize(text):
    return text.split()

with open('text8/text8.train.txt', 'r') as fin:
    text = fin.read()

#select the most common MAX_VOCAB_SIZE words
# and add a UNK word for all uncommon words
text = text.split()
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))

idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)
word_freqs = word_freqs / np.sum(word_freqs)
VOCAB_SIZE = len(idx_to_word)

# create dataset and dataloader
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE - 1) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_word, pos_words, neg_words

dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# defining the model
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        batch_size = input_labels.size(0)

        input_embedding = self.in_embed(input_labels)  # batch_size * embed_size
        pos_embedding = self.out_embed(pos_labels)  # batch_size * (2*context_window) * embed_size
        neg_embedding = self.out_embed(neg_labels)  # batch_size * (2*context_window * K) * embed_size

        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()  # batch_size * (2*context_window)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()  # batch_size * (2*context_window*K)

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size

        loss = log_pos + log_neg

        return -loss

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()

model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(model, dataloader, optimizer):
    model.train()
    for idx, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long().to(device)
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print("Train Epoch: {}, iteration: {}, Loss: {}".format(epoch, idx, loss.item()))

# training phase
for epoch in range(NUM_EPOCHS):
    train_loss = train(model, dataloader, optimizer)

    # save model
    torch.save(model.state_dict(), "skip_gram-model.pth".format(EMBEDDING_SIZE))

# model.load_state_dict(torch.load("eskip_gram-model.pth".format(EMBEDDING_SIZE)))


