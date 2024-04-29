import torch
import torch.nn as nn
import numpy as np
import Extract_PDF
import re
import torch.nn.functional as F
torch.manual_seed(7)
np.random.seed(1)

# Code reference
# https://www.youtube.com/watch?v=bbvr-2hY4mE&t=778s

# Extracting file object
file_object = Extract_PDF.read_data("data/Apocalypse_Now.pdf")

# reading data into data_list, every page of our pdf is a string
text = Extract_PDF.extract_all_data(file_object)  # length = 138, numofPages.


def clean_text(text):
    """function to convert text to lowercase, retain alphabets, replace hyphenation with space"""

    text = text.replace('-', '')
    text = re.sub(r'\d+', '', text)
    text = text.translate({ord(i): None for i in '!"?();´'})
    text = text.replace('--', '')

    return text


Clean_Text = clean_text(text)

chars = tuple(set(Clean_Text))

int2char = dict(enumerate(chars))

char2int = {w: i for i, w in int2char.items()}

# each word in tokens converted to an integer.
encoded = np.array([char2int[char] for char in Clean_Text])


def one_hot_encode(arr, n_labels):

    # Initialize the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot

# encoded contains 158684 characters
# break these into mini batches with some seq_length.
# create input and target tensors for each.
# target is the next character in the sequence


def get_batches(arr, n_seqs, n_steps):

    batch_size = n_seqs * n_steps

    # total number of characters in our batch, in vision, number of images in a batch
    # is batch size, 64, 32 and so on. total # of batches is the length of training set/batch_size.

    n_batches = len(arr)//batch_size  # total num of batches to make from encoded array.

    # following is total characters to keep from encoded array to have full batches which are evenly spaced.

    # number of batches * characters per batch will give the number of characters to slice form our encoded array.
    arr = arr[:n_batches * batch_size]
    # print(arr)
    # print(arr.shape)

    # we will be splitting this array enc into N number of batches each having seq_len number of characters.

    arr = arr.reshape((n_seqs, -1))
    # print(arr)
    # print(arr.shape)

    # sq_len * number of batches, this should be equal to the second dimension of our array.
    # print(40 * 396) # product of this with n_seqs_per_batch gives total number of characters.

    # 10 * 40 window 10 * 15480

    # we now need to split our features

    for n in range(0, arr.shape[1], n_steps):

        x = arr[:, n:n+n_steps]

        y = np.zeros_like(x)

        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]

        yield x, y


batches = get_batches(encoded, 10, 40)


# Define the network.


class CharLSTM(nn.Module):

    def __init__(self, chars, hidden_size, n_layers=2, drop_out=0.5, lr=0.001):

        super(CharLSTM, self).__init__()

        # Set all the hyperparameters of your network

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.drop_out = drop_out
        self.lr = lr

        # set vocabulary and get indices for these
        self.chars = chars
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {w: i for i, w in self.int2char.items()}

        # define the lstm network, this outputs the next char and cell state and hidden state
        self.lstm = nn.LSTM(input_size=len(self.chars), hidden_size=hidden_size, num_layers=n_layers, dropout=drop_out,
                            batch_first=True)
        # add dropout
        self.drop_out = nn.Dropout(drop_out)

        self.fc = nn.Linear(hidden_size, len(chars))

        self.init_weights()

    def forward(self, x, h_0):
        """compute current and hidden units, stack outputs and pass it to linear layer"""

        x, (h, c) = self.lstm(x, h_0)

        x = self.drop_out(x)

        # reshape x: stack the predicted and hidden state outputs
        # cause fully-connected.
        x = x.view(x.size()[0] * x.size()[1], self.hidden_size)

        x = self.fc(x)

        return x, (h, c)

    def predict(self, char, h=None, cuda=False, top_k=None):
        """given a character, predict the next character in the sequence."""

        if cuda:
            self.cuda()
        else:
            self.cpu()

        # Initialize hidden state
        if h is None:
            h = self.init_hidden(1)

        # get the integer of character.
        x = np.array([[self.char2int[char]]])

        # one_hot_encode
        x = one_hot_encode(x, len(self.chars))

        # convet to tensor
        one_hot = torch.from_numpy(x)

        if cuda:
            one_hot = one_hot.cuda()

        # create a tuple of the hidden state
        # this is what LSTM expects
        h = tuple([each.data for each in h])
        out, h = self.forward(one_hot, h)

        # Prob distribution over all the characters
        probs = F.softmax(out, dim=1).data

        # convert back prob to cpu if model was
        # set to gpu

        if cuda:
            probs = probs.cpu()

        # if top number of preds to get
        # wasn't pass, take the distribution over whole character length
        if top_k is None:
            top_ch = np.arange(len(self.chars))
        else:
            probs, top_ch = probs.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        # reduce dims of size 1
        probs = probs.numpy().squeeze()

        char = np.random.choice(top_ch, p=probs / probs.sum())

        return self.int2char[char], h

    def init_weights(self):

        initrange = 0.1

        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs):

        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.hidden_size).zero_(),
                weight.new(self.n_layers, n_seqs, self.hidden_size).zero_())


# Let us start training our network

def train(model, data, epochs=10, n_seqs=10, n_steps=40, lr=0.001, clip=5, val_frac=0.1, cuda=False,
          print_every=10):
    """ model: the model to b trained
        data: the data on which we train
        epochs: number of epochs to train for
        n_seqs" number of sequences in our batch
        n_steps: time step for each sequence
        lr: learning rate
        clip: value used to clip the network gradient to prevent exploding gradeint.
        val_frac: the fraction of data used for validation
        print_every: the number of seconds for which we print out model statistics
    """

    # change model to train mode
    model.train()

    # define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # trin and validation split
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if cuda:
        model.cuda()

    counter = 0
    n_chars = len(model.chars)

    # loop over epochs
    for epoch in range(epochs):

        # initialize hidden layer of the model
        h = model.init_hidden(n_seqs)

        # loop over batches
        for x, y in get_batches(data, n_seqs, n_steps):

            counter += 1

            # one hot encode
            x = one_hot_encode(x, n_chars)

            # convert to tensors
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            # move inputs and targets to cuda

            inputs, targets = inputs.cuda(), targets.cuda()

            # New hidden state being created to prevented backpropogating through the
            # entire history
            h = tuple([each.data for each in h])

            # zero out gradient to prevent accumulation
            model.zero_grad()

            # get output and hidden
            out, h = model.forward(inputs, h)
            loss = criterion(out, targets.view(n_seqs * n_steps).type(torch.cuda.LongTensor))

            # backpropogate loss
            loss.backward()

            # use gradient clipping to prevent exploding gradient
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # take a step in the los surface
            optimizer.step()

            if counter % print_every == 0:

                # initilize hidden state for validation
                val_hidden = model.init_hidden(n_seqs)
                val_losses = []

                for x, y in get_batches(val_data, n_seqs, n_steps):

                    x = one_hot_encode(x, n_chars)

                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    val_hidden = tuple([each.data for each in val_hidden])

                    inputs, targets = x, y

                    if cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    out, val_hidden = model.forward(inputs, val_hidden)

                    val_loss = criterion(out, targets.view(n_seqs * n_steps).type(torch.cuda.LongTensor))

                    val_losses.append(val_loss.item())

                print("Epoch: {}/{}...".format(epoch + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


def sample(model, size, prime='The', top_k=None, cuda=False):
    if cuda:
        model.cuda()
    else:
        model.cpu()

    model.eval()

    # Prime characeters are starting points of some data one can use
    chars = [ch for ch in prime]

    h = model.init_hidden(1)

    for ch in prime:
        char, h = model.predict(ch, h, cuda=cuda, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = model.predict(chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


def main():
    model = CharLSTM(chars, hidden_size=512, n_layers=2)

    n_seqs, n_steps = 128, 100
    train(model, encoded, epochs=25, n_seqs=n_seqs, n_steps=n_steps, lr=0.001, cuda=True, print_every=10)

    print(sample(model, 2000, prime='Horror', top_k=5, cuda=True))


if __name__ == "__main__":
    main()




