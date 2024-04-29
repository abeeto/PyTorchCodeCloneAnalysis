# Hochreiter & Schmidhuber classification task (1997):
# X, X -> Q
# X, Y -> R
# Y, X -> S
# Y, Y -> U

import torch
import torch.nn as nn

from res.sequential_tasks import TemporalOrderExp6aSequence as QRSU
from res.plot_lib import set_default, plot_state, print_colourbar
#%%
# Define the RNN class:
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.f = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.lstm(x)[0] # output features (ignore hidden state)
        x = self.f(x)
        return x
    
    def heatmap(self, x):
        cell_states = []
        hidden_states = []
        state = None
        with torch.no_grad():
            for t in range(x.size(1)):
                state = self.lstm(x[:, [t], :], state)[1]
                cell_states.append(state[1])
                hidden_states.append(state[0])
            c_states = torch.cat(cell_states)
            h_states = torch.cat(hidden_states)
        return c_states, h_states
#%%
# Define a model and the task:

# Define difficulty of the task:        
difficulty = QRSU.DifficultyLevel.NORMAL # EASY, NORMAL, MODERATE or HARD
batch_size = 32

# Create data generators:
train_data = QRSU.get_predefined_generator(difficulty, batch_size)
test_data  = QRSU.get_predefined_generator(difficulty, batch_size)
    
# Define model parameters:
input_size  = train_data.n_symbols
hidden_size = 12 # 64 for DifficultyLevel.MODERATE
output_size = train_data.n_classes

# Training on GPU:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define a model:
net = LSTM(input_size, hidden_size, output_size).to(device)

# Optimizer and an objective function:
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-3)

# An objective function:
criterion = torch.nn.CrossEntropyLoss()

# Additional parameters:
epochs = 10
#%%
# Train the network:

for epoch in range(epochs):
    
    # Number of correctly classified sequences:
    num_correct = 0
    
    for batch_id in range(len(train_data)):
        
        # Unpack the data:
        data_X, data_Y = train_data[batch_id]
        
        # Convert the data into tensors and send to GPU:
        data_X = torch.from_numpy(data_X).float().to(device)
        data_Y = torch.from_numpy(data_Y).long().to(device)
        
        # Compute the output:
        output = net(data_X)
        
        # Remove padding:
        output = output[:, -1, :]
        
        # Determine real labels for batches:
        data_Y = data_Y.argmax(dim=1)
        
        # Compute training loss:
        loss = criterion(output, data_Y)
        
        # Reset the gradients:
        optimizer.zero_grad()
        
        # Compute the gradients:
        loss.backward()
        
        # Update the parameters:
        optimizer.step()
        
        # Determine predicted labels for batches:
        pred_Y = output.argmax(dim=1)
        
        # Update the number of correctly classified sequences:
        num_correct += (pred_Y == data_Y).sum().item()
                
        # Display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss.item()))

print('Finished Training')
#%%
# Test the network on the test data:

# Number of correctly classified sequences:
num_correct = 0

# Disable gradient calculations during inference:
with torch.no_grad():
    for batch_id in range(len(test_data)):
        
        # Unpack the data:
        data_X, data_Y = test_data[batch_id]
        
        # Convert the data into tensors and send to GPU:
        data_X = torch.from_numpy(data_X).float().to(device)
        data_Y = torch.from_numpy(data_Y).long().to(device)

        # Compute predictions:
        output = net(data_X)
        
        # Remove padding:
        output = output[:, -1, :]

        # Determine real labels for batches:
        data_Y = data_Y.argmax(dim=1)
        
        # Compute test loss:
        loss = criterion(output, data_Y)

        # Determine predicted labels for batches:
        pred_Y = output.argmax(dim=1)
        
        # Update the number of correctly classified sequences:
        num_correct += (pred_Y == data_Y).sum().item()
                
        # Compute accuracy:
        accuracy = float(num_correct) / (len(test_data) * test_data.batch_size) * 100
        
# Display the testing loss:
print("loss = {:.6f}, accuracy = {:2.2f}%".format(loss.item(), accuracy))

print('Finished Testing')
#%%
set_default()

# Get hidden (H) and cell (C) batch state given a batch input (X)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#net.eval()
with torch.no_grad():
    data = test_data[0][0]
    X = torch.from_numpy(data).float().to(device)
    cell_states, hidden_states = net.heatmap(X)
#%%
# Use Jupyter Notebook to get the following plots:
    
# Legend (color range):
html_obj = print_colourbar()

# Heatmap for the cell (memory) states:
plot_state(X.cpu(), cell_states, b=9, decoder=test_data.decode_x)

# Heatmap of the hidden states:
plot_state(X.cpu(), hidden_states, b=9, decoder=test_data.decode_x)
#%%