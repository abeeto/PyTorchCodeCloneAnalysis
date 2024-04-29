# -------------- Text generation ----------------
# ---------- Character Level Encoding -----------
# --------------    Vanilla RNN     -------------

# hidden(t) = tanh ( Weight(t) * hidden(t-1) + Weight(input) * Input(t) )
# output(t) = Weight(output) * hidden(t)

# ------------ Implementation Steps -----------
# 1 creating vocab
# 2 padding
# 3 splitting into input/labels
# 4 one-hot encoding
# 5 model defining
# 6 model training
# 7 model evaluation

import torch
from torch import nn
import numpy as np

# 1 ---------------------------------------------
text = ["hey how are you", "good i am fine", "have a nice day"]
chars = set(" ".join(text)) # joining the documents and getting unique characters i.e. vocabulary
int2char = dict(enumerate(chars)) # dictionary : mapping of character on an integer

char2int = {} # dictionary: mapping of integers on a character for reversing the encoding
for k,v in int2char.items():
    char2int.update({v:k})

print(char2int)
# 2 ---------------------------------------------
# padding w.r.t the longest string
longest = max(text, key=len)
max_len = len(longest)

for i in range(len(text)):      # all documents in text
    while len(text[i])<max_len: # append whitespace at the end of the doc till it reaches max_len
        text[i] += " "
print("After padding text: ",text)
# 3 ---------------------------------------------
# input: last character removed
# label/ground truth: first character removed (one time-step ahead of input data)
input_seq = []
label_seq = []

for i in range(len(text)):
    input_seq.append(text[i][:-1])
    label_seq.append(text[i][1:])
print("Input: ",input_seq)
print("Label: ",label_seq)

# 4 ---------------------------------------------
# First integer encoding then one-hot-encoding
# integer encoding the input and label
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    label_seq[i] = [char2int[character] for character in label_seq[i]]
print(input_seq)
print(label_seq)

# dict_size :   dictionary length - determines the length of one-hot vectors
# seq_len   :   length of sequences being fed to the model which is fixed: maxlength-1 as last char got removed
# batch_size:   the number of sentences in one batch

dict_size = len(char2int) # 17
seq_len = max_len-1       # 14
batch_size = len(text)    # 3

def hot_encoder(input_sequence, batch_size, seq_len, dict_size):
    """ creates arrays of zeroes for each character in a sequence
    and replaces the corresponding character-index with 1 i.e. makes
    it "hot".
    """
    # multi-dim array of zeros with desired output size
    # shape = 3,14,17
    # every doc is 14 char long. Each char is a 17 items long one-hot vector as dict_size is 17
    onehot_data = np.zeros(shape=(batch_size, seq_len, dict_size))

    # replacing the 0 with 1 for each character to make its one-hot vector
    for batch_item in range(batch_size):
        for sequence_item in range(seq_len):
            onehot_data[batch_item, sequence_item, input_sequence[batch_item][sequence_item] ] = 1

    print("Shape of one-hot encoded data: ", onehot_data.shape)
    return onehot_data

input_seq = hot_encoder(input_seq,batch_size, seq_len, dict_size)
print("Input data after being shaped and one-hot:\n",input_seq)

# From np array Tensor
input_seq = torch.from_numpy(input_seq)
label_seq = torch.Tensor(label_seq)
print(label_seq)

# 5 ---------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cpu")


# Model: 1 layer of RNN followed by a fully connected layer.
# This fully connected layer is responsible for converting the RNN output to our desired output shape.
# The forward function is executed sequentially, therefore we'll have to pass the inputs and zero-initialized hidden
# state through the RNN layer first,before passing the RNN outputs to the fully connected layer.
# init_hidden(): Initializes the hidden state. Creates a tensor of zeros in the shape of our hidden states.
class RnnModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(RnnModel, self).__init__()
        # parameter/data-member defining
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # layer defining
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0) # size of input batch along axis=0
        # initializing the hidden state for first input using init_hidden method defined below
        hidden = self.init_hidden(batch_size)

        # Input + hidden_state to RNN to generate output
        out, hidden = self.rnn(x, hidden)

        # Reshaping the output so it can be fit to the fully connected layer
        # contiguous() -> returns a contiguous tensor. good for performance.
        # output and labels/targets both should be contiguous for computing the loss
        # view() cannot be applied to a dis-contiguous tensor
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)

        return out,hidden

    def init_hidden(self, batch_size):
        # Generates the first hidden layer of zeros to be used in the forward pass.
        # the tesor holding the hidden state will be sent to the device specified earlier
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden

# Instantiating the Model
model = RnnModel(input_size=dict_size, output_size=dict_size, hidden_size=12, n_layers=1)
model.to(device)

# Hyper-parameters
n_epochs = 100
lr =0.01

# Loss, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# 6 ---------------------------------------------
# Training
for epoch in range(1, n_epochs+1):
    optimizer.zero_grad()   # clearing existing gradients from previous epoch
    input_seq.to(device)
    output, hidden = model(input_seq.float()) # Error: Expected object of scalar type float but got scalar type Double
    loss = criterion(output, label_seq.view(-1).long())
    loss.backward() # backpropagation gradient calculation
    optimizer.step() # weights update

    if epoch%10 == 0:
        print('Epoch: {}/{}...............'.format(epoch, n_epochs), end=' ')
        print('Loss: {:.4f}'.format(loss.item()))

# 7 ---------------------------------------------
# Sample takes the string, makes one-hot tensor of each character using one_hotter_char,
# and passes it to predict for next character prediction

# one-hot for training is not working on the single character. So this one does it separately
def one_hotter_char(char, batch_size, seq_len, dict_size):
    """makes a single input character one-hot-encoded and returns a tensor"""
    onehot_data = np.zeros(shape=(batch_size, seq_len, dict_size))
    char = char2int[char]
    print("Integer Encoded Character is: ",char)
    onehot_data[0][0][char] = 1
    print("One-hot Character is: ",onehot_data)
    print(onehot_data.shape)
    # making it a tensor
    onehot_data = torch.from_numpy(onehot_data).float() # .float() is required otherwise Error: Expected object of scalar type Double but got scalar type Float
    onehot_data.to('cpu')
    print(type(onehot_data))
    return onehot_data


def predict(model, char_tensor):
    """Takes model and a one-hot encoded tensor of a character, and predicts the next character
    and returns it along with the hidden state."""

    out, hidden = model(char_tensor)

    probability = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class having highest score from the output
    char = torch.max(probability, dim=0)[1].item()
    char = int2char[char] # converting from int to character

    return char, hidden


def sample(model, out_len, batch_size, seq_len, dict_size, start_txt="hey"):
    """Takes the desired output length and input characters as input and returns the produced sentence."""
    model.eval() # evaluation mode of the model
    start_txt = start_txt.lower()
    chars = [c for c in start_txt]
    size = out_len - len(chars) # assuming that output_length is always bigger than input characters

    for i in range(size):
        char_tensor = one_hotter_char(chars[i], batch_size, seq_len, dict_size)
        predicted_char, h = predict(model, char_tensor)
        chars.append(predicted_char)

    return ''.join(chars)


one_hotter_char("g", batch_size=1, seq_len=1, dict_size=dict_size)

generated_txt = sample(model, 15, batch_size=1, seq_len=1, dict_size=dict_size, start_txt="good")
print(generated_txt)
