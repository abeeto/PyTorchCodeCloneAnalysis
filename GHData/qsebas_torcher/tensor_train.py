#!/usr/bin/env python
import pickle

from tqdm import tqdm
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import numpy as np

from glob import glob
import random
import os

def load_imdb_data(num_words, sequence_length, test_size=0.25, oov_token=None):
    # read reviews
    reviews = []
    with open("data/reviews.txt") as f:
        for review in f:
            reviews.append(review)

    labels = []
    with open("data/labels.txt") as f:
        for label in f:
            label = label.strip()
            labels.append(label)
    # tokenize the dataset corpus, delete uncommon words such as names, etc.
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(reviews)
    X = tokenizer.texts_to_sequences(reviews)
    X, y = np.array(X), np.array(labels)
    # pad sequences with 0's
    X = pad_sequences(X, maxlen=sequence_length)
    # convert labels to one-hot encoded
    y = to_categorical(y)
    # split data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    data = {}
    data["X_train"] = X_train
    data["X_test"]= X_test
    data["y_train"] = y_train
    data["y_test"] = y_test
    data["tokenizer"] = tokenizer
    data["int2label"] =  {0: "negative", 1: "positive"}
    data["label2int"] = {"negative": 0, "positive": 1}

    # saving
    with open(os.path.join ("results", 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data

def get_embedding_vectors(word_index, embedding_size=100):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_size))
    with open(f"data/glove.6B.{embedding_size}d.txt", encoding="utf8") as f:
        for line in tqdm(f, "Reading GloVe"):
            values = line.split()
            # get the word as the first word in the line
            word = values[0]
            if word in word_index:
                idx = word_index[word]
                # get the vectors as the remaining values in the line
                embedding_matrix[idx] = np.array(values[1:], dtype="float32")
    return embedding_matrix

def create_model(word_index, units=128, n_layers=1, cell=LSTM, bidirectional=False,
                embedding_size=100, sequence_length=100, dropout=0.3,
                loss="categorical_crossentropy", optimizer="adam",
                output_length=2):
    """Constructs a RNN model given its parameters"""
    embedding_matrix = get_embedding_vectors(word_index, embedding_size)
    model = Sequential()
    # add the embedding layer
    model.add(Embedding(len(word_index) + 1,
              embedding_size,
              weights=[embedding_matrix],
              trainable=False,
              input_length=sequence_length))
    for i in range(n_layers):
        if i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # first layer or hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(Dense(output_length, activation="softmax"))
    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


# max number of words in each sentence
SEQUENCE_LENGTH = 300
# N-Dimensional GloVe embedding vectors
EMBEDDING_SIZE = 300
# number of words to use, discarding the rest
N_WORDS = 10000
# out of vocabulary token
OOV_TOKEN = None
# 30% testing set, 70% training set
TEST_SIZE = 0.3
# number of CELL layers
N_LAYERS = 1
# the RNN cell to use, LSTM in this case
RNN_CELL = LSTM
# whether it's a bidirectional RNN
IS_BIDIRECTIONAL = False
# number of units (RNN_CELL ,nodes) in each layer
UNITS = 128
# dropout rate
DROPOUT = 0.4
### Training parameters
LOSS = "categorical_crossentropy"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 6

def get_model_name(dataset_name):
    # construct the unique model name
    model_name = f"{dataset_name}-{RNN_CELL.__name__}-seq-{SEQUENCE_LENGTH}-em-{EMBEDDING_SIZE}-w-{N_WORDS}-layers-{N_LAYERS}-units-{UNITS}-opt-{OPTIMIZER}-BS-{BATCH_SIZE}-d-{DROPOUT}"
    if IS_BIDIRECTIONAL:
        # add 'bid' str if bidirectional
        model_name = "bid-" + model_name
    if OOV_TOKEN:
        # add 'oov' str if OOV token is specified
        model_name += "-oov"
    return model_name

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")
# dataset name, IMDB movie reviews dataset
dataset_name = "imdb"
# get the unique model name based on hyper parameters on parameters.py
model_name = get_model_name(dataset_name)
# load the data
data = load_imdb_data(N_WORDS, SEQUENCE_LENGTH, TEST_SIZE, oov_token=OOV_TOKEN)
# construct the model
model = create_model(data["tokenizer"].word_index, units=UNITS, n_layers=N_LAYERS,
                    cell=RNN_CELL, bidirectional=IS_BIDIRECTIONAL, embedding_size=EMBEDDING_SIZE,
                    sequence_length=SEQUENCE_LENGTH, dropout=DROPOUT,
                    loss=LOSS, optimizer=OPTIMIZER, output_length=data["y_train"][0].shape[0])
model.summary()
# using tensorboard on 'logs' folder
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
# start training
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[tensorboard],
                    verbose=1)
# save the resulting model into 'results' folder
model.save(os.path.join("results", model_name) + ".h5")

