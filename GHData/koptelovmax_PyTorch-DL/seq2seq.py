# Signal echoing
#  Input -- sequence, output -- same sequence shifted over time
#  Example:
#  In:  1 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 1 0 0 1 ... 
#  Out: 0 0 0 1 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 1 ...

import torch
import torch.nn as nn

from res.sequential_tasks import EchoData
#%%
# Define the RNN class:
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.f = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        x, h = self.rnn(x, h)
        x = self.f(x)
        return x, h
#%%
# Define a model and the task:

# Define task parameters:        
batch_size = 5
echo_step = 3
series_length = 20000
trunc_length = 20

# Create data generators:
train_data = EchoData(
    echo_step=echo_step,
    batch_size=batch_size,
    series_length=series_length,
    truncated_length=trunc_length,
)

test_data = EchoData(
    echo_step=echo_step,
    batch_size=batch_size,
    series_length=series_length,
    truncated_length=trunc_length,
)
    
# Define model parameters:
input_size = 1
hidden_size = 4
output_size = 1

# Training on GPU:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define a model:
net = RNN(input_size, hidden_size, output_size).to(device)

# Optimizer and an objective function:
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-3)

# An objective function:
criterion = torch.nn.BCEWithLogitsLoss()

# Additional parameters:
epochs = 5
#%%
# Train the network:

# No previous hidden layer at the beginning:
hidden_state = None

for epoch in range(epochs):
    
    # Number of correctly predicted batches:
    num_correct = 0
    
    for batch_id in range(len(train_data)):
        
        # Unpack the data:
        data_X, data_Y = train_data[batch_id]
        
        # Convert the data into tensors and send to GPU:
        data_X = torch.from_numpy(data_X).float().to(device)
        data_Y = torch.from_numpy(data_Y).float().to(device)

        # Compute the output (taking into account hidden layer of previous batch):
        output, hidden_state = net(data_X, hidden_state)
        
        # Detach the hidden layer (otherwise backpropagation would go until the start of data)
        hidden_state.detach_()
        
        # Compute training loss:
        loss = criterion(output, data_Y)
        
        # Reset the gradients:
        optimizer.zero_grad()
        
        # Compute the gradients:
        loss.backward()
        
        # Update the parameters:
        optimizer.step()
        
        # Determine predicted labels for batches:
        pred_Y = (torch.sigmoid(output) > 0.5)
        
        # Update the number of correctly classified sequences:
        num_correct += (pred_Y == data_Y.byte()).int().sum().item()
        
        # Compute train accuracy:
        train_acc = num_correct / float(len(train_data))
        
    # Display the epoch, training loss and train accuracy:
    print("epoch : {}/{}, loss = {:.6f}, accuracy = {:2.2f}%".format(epoch + 1, epochs, loss.item(), train_acc))

print('Finished Training')
#%%
# Test the network on the test data:

# Number of correctly predicted batches:
num_correct = 0

# Disable gradient calculations during inference:
with torch.no_grad():
    for batch_id in range(len(test_data)):
        
        # Unpack the data:
        data_X, data_Y = test_data[batch_id]
        
        # Convert the data into tensors and send to GPU:
        data_X = torch.from_numpy(data_X).float().to(device)
        data_Y = torch.from_numpy(data_Y).float().to(device)

        # Compute predictions:
        output, hidden_state = net(data_X, hidden_state)
        
        # Determine predicted labels for batches:
        pred_Y = (torch.sigmoid(output) > 0.5)
        
        # Update the number of correctly classified sequences:
        num_correct += (pred_Y == data_Y.byte()).int().sum().item()
                
        # Compute accuracy:
        test_acc = float(num_correct) / float(len(test_data))
        
# Display the test accuracy:
print("Test accuracy = {:.1f}%".format(test_acc))

print('Finished Testing')
#%%
# Test how it works:
X = torch.empty(1, 100, 1).random_(2).to(device)
output,_ = net(X, None)
print('Input:', ' '.join(str(i) for i in (X.view(1, -1).byte().tolist()[0])))
print('Output:', ' '.join(str(i) for i in ((output > 0).view(1, -1).byte().tolist()[0])))
#%%