import helper
import torch
import numpy as np
import torch.nn as nn
from tv_scripts_rnn import RNN
from torch.utils.data import TensorDataset, DataLoader


int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
print(len(int_text))

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a CPU to train your neural network.')


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """

    batch_size_total = batch_size * sequence_length
    n_batches = len(words) // batch_size_total

    words = words[:n_batches * batch_size_total]

    feature_tensors = []
    target_tensors = []

    for n in range(n_batches):
        for i in range(batch_size):
            feature_tensors.append(words[(n*batch_size_total)+i:(n*batch_size_total)+i+sequence_length])
            target_tensors.append(words[(n*batch_size_total)+i+sequence_length])

    train_data = TensorDataset(torch.tensor(feature_tensors), torch.tensor(target_tensors))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    # return a dataloader
    return train_loader


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param rnn: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # move data to GPU, if available
    if (train_on_gpu):
        inp, target = inp.cuda(), target.cuda()
    
    # perform backpropagation and optimization
    hidden = tuple([each.data for each in hidden])

    rnn.zero_grad()

    output, hidden = rnn(inp, hidden)

    loss = criterion(output.squeeze(), target.long())
    loss.backward()

    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden


# Data params
# Sequence Length
sequence_length = 10 # of words in a sequence
# Batch Size
batch_size = 64
train_loader = batch_data(int_text, sequence_length, batch_size)


def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn


# Training parameters
# Number of Epochs
num_epochs = 16
# Learning Rate
learning_rate = 0.001

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = len(vocab_to_int)
# Embedding Dimension
embedding_dim = 600
# Hidden Dimension
hidden_dim = 512
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 10

# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('trained_rnn', trained_rnn)
print('Model Trained and Saved')