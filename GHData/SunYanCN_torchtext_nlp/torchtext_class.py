import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import os
import pandas as pd
import numpy as np
import torch
import spacy
import tqdm
import random
from torchtext.vocab import GloVe
from torchtext.vocab import Vectors
from torchtext.data import Field, TabularDataset
from torchtext.data import Iterator, BucketIterator
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.nn import init



val_ratio = 0.2
NLP = spacy.load('en')
MAX_CHARS = 20000

def seed_everything(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def prepare_csv():
    df_train = pd.read_csv('data/train.csv')
    df_train["comment_text"] = df_train["comment_text"].str.replace("\n", " ")
    idx = np.arange(df_train.shape[0])

    np.random.shuffle(idx)
    val_size = int(len(idx) * val_ratio)
    df_train.iloc[idx[val_size:], :].to_csv("cache/train.csv", index=False)
    df_train.iloc[idx[:val_size], :].to_csv("cache/val.csv", index=False)

    df_test = pd.read_csv('data/test.csv')
    df_test["comment_text"] = df_test["comment_text"].str.replace("\n", " ")
    df_test.to_csv("cache/test.csv", index=False)


def tokenizer(comment):
    fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    trans_map = str.maketrans(fileters, " " * len(fileters))
    comment = comment.translate(trans_map)
    if len(comment) > MAX_CHARS:
        comment = comment[:MAX_CHARS]
    return [x.text for x in NLP.tokenizer(comment) if x.text != " "]



def get_dataset(fix_length=100, lower=False, vectors=None):
    if vectors is not None:
        lower = True
    logging.info("预处理 csv......")
    prepare_csv()

    TEXT = Field(sequential=True, fix_length=fix_length, tokenize=tokenizer, pad_first=True, lower=lower)
    LABEL = Field(sequential=False, use_vocab=False)

    train_datafields = [("id", None),
                        ("comment_text", TEXT), ("toxic", LABEL),
                        ("severe_toxic", LABEL), ("threat", LABEL),
                        ("obscene", LABEL), ("insult", LABEL),
                        ("identity_hate", LABEL)]

    logging.info("读取 train.csv......")
    train, val = TabularDataset.splits(path='cache', train='train.csv', validation="val.csv",
                                format='csv',
                                skip_header=True,
                                fields=train_datafields
                                )
    logging.info("读取 test.csv......")
    test = TabularDataset(path='cache/test.csv',
                          format='csv',
                          skip_header=True,
                          fields=[('id', None), ('comment_text', TEXT)])

    logging.info('读取glove词向量......')
    # vectors = GloVe(name='6B', dim=300) #会下载词向量
    #读取本地词向量
    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name='/home/sunyan/quora/input/embeddings/glove.840B.300d.txt',cache=cache,max_vectors =200000)
    vectors.unk_init = init.xavier_uniform_

    logging.info('构建词表......')
    TEXT.build_vocab(train, test, max_size=20000, min_freq=50, vectors=vectors)

    print(TEXT.vocab.freqs.most_common(10))

    logging.info("预处理结束！")

    return (train, val, test), TEXT


class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)

            if self.y_vars is not None:
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)


def get_iterator(dataset, device, batch_size, shuffle=True, repeat=False):
    train, val, test = dataset

    train_iter, val_iter = BucketIterator.splits(
        (train, val),
        batch_size=batch_size,
        device=device,
        sort_key=lambda x: len(x.comment_text),
        sort_within_batch=False,
        shuffle=shuffle,
        repeat=repeat)

    test_iter = Iterator(test,
                         batch_size=batch_size,
                         device=device,
                         sort_within_batch=False,
                         repeat=repeat,
                         sort=False)

    train_dl = BatchWrapper(train_iter, "comment_text",
                            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    valid_dl = BatchWrapper(val_iter, "comment_text",
                            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    test_dl = BatchWrapper(test_iter, "comment_text", None)

    return train_dl, valid_dl, test_dl


class SimpleBiLSTM(nn.Module):

    def __init__(self, hidden_size, lin_size, max_features, emb_dim, embedding_matrix):
        super(SimpleBiLSTM, self).__init__()

        # Initialize some parameters for your model
        self.hidden_size = hidden_size
        drp = 0.1

        # Layer 1: Word2Vec Embeddings.
        self.embedding = nn.Embedding(max_features, emb_dim)
        self.embedding.weight.data.copy_(embedding_matrix)

        # Layer 2: Dropout1D(0.1)
        self.embedding_dropout = nn.Dropout2d(0.1)

        # Layer 3: Bidirectional CuDNNLSTM
        self.lstm = nn.LSTM(emb_dim, hidden_size, bidirectional=True, batch_first=True)

        # Layer 4: Bidirectional CuDNNGRU
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        # Layer 7: A dense layer
        self.linear = nn.Linear(hidden_size*6, lin_size)
        self.relu = nn.ReLU()

        # Layer 8: A dropout layer
        self.dropout = nn.Dropout(drp)

        # Layer 9: Output dense layer with one output for our Binary Classification problem.
        self.out = nn.Linear(lin_size, 6)


    def forward(self, x):
        # x: [max_len,batch_size]
        # embeddings: [max_len,batch_size,emb_dim]
        embeddings = self.embedding_dropout(self.embedding(x))
        # embeddings: [batch_size,max_len,emb_dim]
        embeddings = embeddings.permute(1,0,2)
        #h_lstm: [batch_size, max_len, hidden_size]
        h_lstm, _ = self.lstm(embeddings)
        #h_gru: [batch_size, max_len, hidden_size]
        #hh_gru: [num_layers,batch_size, hidden_size]
        h_gru, hh_gru = self.gru(h_lstm)
        # hh_gru: [batch_size, hidden_size×2]
        hh_gru = hh_gru.view(-1, 2 * self.hidden_size)

        #avg_pool： [batch_size, hidden_size]
        avg_pool = torch.mean(h_gru, 1)
        #max_pool： [batch_size, hidden_size]
        max_pool, _ = torch.max(h_gru, 1)

        #hh_gru:[128,512] avg_pool:[100,512] max_pool:[100,512]
        conc = torch.cat((hh_gru, avg_pool, max_pool), 1)

        # passing conc through linear and relu ops
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        # conc:[100,6]
        out = self.out(conc)
        # return the final output
        return out


if __name__ == '__main__':

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    seed_everything()

    dataset, TEXT = get_dataset()
    train, val, test = dataset
    train_dl, valid_dl, test_dl = get_iterator(dataset,device,batch_size=256)

    embedding_matrix = TEXT.vocab.vectors
    vocab_size = len(TEXT.vocab)

    model = SimpleBiLSTM(hidden_size=256,
                         lin_size=6,
                         max_features=vocab_size,
                         emb_dim=300,
                         embedding_matrix=embedding_matrix,
                         )
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    epochs = 10
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_corrects = 0
        model.train()  # turn on training mode
        # x:[100,128] y:[128,6]
        for x, y in tqdm.tqdm(train_dl):  # thanks to our wrapper, we can intuitively iterate over our data!
            opt.zero_grad()
            preds = model(x)
            loss = loss_func(preds, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(train)

        # calculate the validation loss for this epoch
        val_loss = 0.0
        model.eval()  # turn on evaluation mode
        for x, y in valid_dl:
            preds = model(x)
            loss = loss_func(preds, y)
            val_loss += loss.item() * x.size(0)

        val_loss /= len(val)
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
