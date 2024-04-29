import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import matplotlib.pyplot as plt

class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        #This creates the layers
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(11, 10),
            nn.ReLU(),
            nn.Linear(10, 7),
            nn.ReLU(),
            nn.Linear(7, 1)
        )

    def forward(self, x):
        #forward function must be defined, so we give it the layers we created above
        return self.linear_relu_stack(x)

#this function is used to initialize the weights to a set value to reduce randomness
def init_weights(layer):
    if type(layer) == nn.Linear:
        layer.weight.data.fill_(0.0)

model = ExampleNet()
#model.apply(init_weights)
num_epochs = 1000
optimizer = optim.SGD(model.parameters(), lr=.00001) #learning rates higher than this tend to converge to local minima after first epoch
testing_differences = -1
curr_epoch = 0

#variables for pyplots
testing_error_y_vals = []
training_error_y_vals = []
x_vals_training = []
x_vals_testing = []

convergence_counter = 0
divergence_counter = 0

plt.xlim(1, 50)
plt.ylim(0, 8)
plt.title("Error on validation set")
plt.xlabel("Epochs")
plt.ylabel("Average error on samples")

#Training
for outer_loop in range(num_epochs):
    if outer_loop % 10 == 0:
        #for group in optimizer.param_groups:
        #    group['lr'] = group['lr'] - (group['lr']/10) #lowers weights slowly over time
        print(outer_loop)
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] - (group['lr']/100)
    f = open("training.csv")
    f.readline()
    sum_differences = 0
    num_samples = 0
    for line in f:
        line = line.split(";")
        line = list(map(float, line))
        target = [line[-1]] #target must be in list in order to be converted to tensor
        target = torch.FloatTensor(target) 
        line = line[:-1]
        #line = [float(i)/max(line) for i in line] #normalize the data
        output = model(torch.FloatTensor(line))
        optimizer.zero_grad()
        loss_func = nn.MSELoss()
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        sum_differences += abs(float(target.item() - output))
        num_samples += 1
    training_error_y_vals.append(float(sum_differences)/num_samples)
    x_vals_training.append(curr_epoch)

    #Overfitting prevention here
    divergence_threshold = .001 #if (new average error) - (prev average error) > divergence_threshold
    convergence_threshold = .0005 #if abs(new average error - prev average error) < convergence_threshold 
    f = open("validation.csv")
    f.readline()
    sum_differences = 0
    num_samples = 0
    for line in f:
        line = line.split(";")
        line = list(map(float, line))
        target = [line[-1]]
        target = torch.FloatTensor(target)
        line = line[:-1]
        #line = [float(i)/max(line) for i in line] 
        output = model(torch.FloatTensor(line))
        sum_differences += abs(float(target.item() - output))
        num_samples += 1
    exit_code = -1

    #checking for convergence and divergence
    if testing_differences == -1: #if it is first iteration
        testing_differences = float(sum_differences)/num_samples
    elif float(sum_differences)/num_samples <= testing_differences: 
        divergence_counter = 0
        if testing_differences - float(sum_differences)/num_samples < convergence_threshold:
            convergence_counter += 1
            if convergence_counter == 10:
                exit_code = 1
                break
        else:
            convergence_counter = 0
            testing_differences = float(sum_differences)/num_samples
    else:
        if float(sum_differences)/num_samples - testing_differences > divergence_threshold:
            divergence_counter += 1
            if divergence_counter == 5:
                exit_code = 0
                break
        else:
            divergence_counter = 0

    curr_epoch += 1

    testing_error_y_vals.append(float(sum_differences)/num_samples)
    x_vals_testing.append(curr_epoch)

    if(max(x_vals_testing) > 49):
        plt.xlim(1, max(x_vals_testing) + 10)
    if(float(sum_differences)/num_samples > 1):
        plt.ylim(0,float(sum_differences)/num_samples * 1.5)
    else:
        plt.ylim(float(sum_differences)/num_samples * float(sum_differences)/num_samples,float(sum_differences)/num_samples * 1.5)
    plt.plot(x_vals_testing, testing_error_y_vals)
    plt.pause(.01)

if exit_code == -1:
    print("Training stopped after " + str(curr_epoch) + " epochs from reaching max epochs.")
elif exit_code == 0:
    print("Training stopped after " + str(curr_epoch) + " epochs from achieving convergence.")
elif exit_code == 1:
    print("Training stopped after " + str(curr_epoch) + " epochs from overfitting prevention.")

f.close()

#Testing
f = open("testing.csv")
f.readline()
sum_differences = 0
num_samples = 0
for line in f:
    line = line.split(";")
    line = list(map(float, line))
    target = [line[-1]]
    target = torch.FloatTensor(target)
    line = line[:-1]
    #line = [float(i)/max(line) for i in line] #this does normalization
    output = model(torch.FloatTensor(line))
    sum_differences += abs(float(target.item() - output))
    num_samples += 1
print("Testing result: ", end="")
print(float(sum_differences)/num_samples)

#Look at this graph
plt.plot(x_vals_testing, testing_error_y_vals)
plt.savefig("testing_differences_by_epoch.png")
plt.close()
plt.plot(x_vals_training, training_error_y_vals)
plt.savefig("training_differences_by_epoch.png")