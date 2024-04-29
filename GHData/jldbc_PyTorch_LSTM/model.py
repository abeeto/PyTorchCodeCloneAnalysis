import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Hyperparameters
"""
n_hidden = 256
n_input = X.shape[0] #it's probably this. verify once data is vectorized
n_layers = 1
num_epochs = 10 #increase this once the model is working
lr = 0.01  #initial learning rate
batch_size = 32 #num observations per batch
seq_len = 80
save_model = True
model_name = "PyTorch_LSTM_Model"


"""
Load in data:

* Read in text file 
* Break into matrix of shape (n_sequences x seq length x num_characters)
* Target = (n_sequences x num_characters) matrix  (i.e. the character that follows each sequence)
* Break X and target into train, validation, and test sets 
"""
#
#
#
n_characters = float('inf')  #find number of characters in the data. this is needed for dimensionality of matrices + calling the loss function


"""
Build the model
"""
def init_weights():
	pass

def create_model(n_input, n_hidden, n_layers):
	w = create_weights()
	rnn = nn.LSTM(n_input, n_hidden, n_layers, bias=True)
	return rnn

def feed_forward(x_input, hidden):
	rnn = create_model(n_input, n_hidden, n_layers)
	output, hidden = rnn(x_input, hidden)
	return output, hidden

model = create_model(n_input, n_hidden, n_layers)


"""
Training functions
"""
criterion = nn.CrossEntropyLoss()

def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    Source: https://github.com/pytorch/examples/word_language_model
    """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def next_batch(x_data, target, idx, evaluation=False):
	x_data = Variable(x_data[idx:idx+batch_size, :], volatile=evaluation) #observations from start_index to start_index + batch_size
    target = Variable(target[idx:idx+batch_size, :])
    return x_data, target

def training_step():
	"""
	a training pass through the data in multiple batches
	"""
	total_loss = 0
	hidden = model.init_hidden(eval_batch_size)  #still need to create this function 
	#iterate over batches
	batch_num = 0
	for idx in range(0, train_data.shape[0], batch_size):
		X, target = next_batch(train_data, idx)
		hidden = repackage_hidden(hidden)
		model.zero_grad()
		out, hidden = model(X, hidden)
		loss = criterion(out.view(-1, n_characters), target)
		loss.backward()
		total_loss += loss.data 
		batch_num += 1

def get_loss():
	pass

"""
Now train the model
"""
for epoch in range(num_epochs):
	training_step()
	loss = get_loss(validation_X, validation_target) #get loss out of sample with this epoch's trained weights

loss = get_loss(test_X, test_target)


#save the model
if save_model == True:
    with open(model_name, 'wb') as f:
        torch.save(model, f)


