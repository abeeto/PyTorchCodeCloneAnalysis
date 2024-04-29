'''
sigmoid의 문제점: Gradient vanishing
sigmoid를 대체할 수 있는 activation function
  - ReLU
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001   # lr을 0.1로 했더니 정확도 40% 정도밖에 안 나옴.
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

# set model
linear1 = nn.Linear(28*28, 256, bias=True).to(device)
linear2 = nn.Linear(256, 256, bias=True).to(device)
linear3 = nn.Linear(256, 10, bias=True).to(device)
relu = nn.ReLU()

nn.init.normal_(linear1.weight)      # 가중치 초기화
nn.init.normal_(linear2.weight)      # 가중치 초기화
nn.init.normal_(linear3.weight)      # 가중치 초기화

model = nn.Sequential(linear1, relu, linear2, relu, linear3).to(device)

# define optimizer
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

nb_epochs = 15
for epoch in range(nb_epochs):
    avg_cost = 0
    batches = len(data_loader)
    
    for X, Y in data_loader:
        X = X.view(-1, 28*28).to(device)

        # hypothesis
        hypothesis = model(X)
        # cost
        cost = F.cross_entropy(hypothesis, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / batches

    print(f'epoch: {epoch+1:3d}/{nb_epochs} Cost: {avg_cost:.6f}')
print('Learning Finished')
        

# evaluation
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    # hypothesis
    hypothesis = model(X_test)
    # accuracy
    isCorrect = torch.argmax(hypothesis, dim=-1) == Y_test
    accuracy = torch.mean(isCorrect.float())

    print(f'Accuracy: {accuracy*100:.3f}%')

    for _ in range(10):
        # visualize
        r = random.randint(0, len(mnist_test) - 1)
        X_single_data = mnist_test.test_data[r:r+1].view(-1, 28*28).float().to(device)
        Y_single_data = mnist_test.test_labels[r:r+1].to(device)

        print('Label: ', Y_single_data.item())
        single_hypothesis = model(X_single_data)
        print('Prediction: ', torch.argmax(single_hypothesis, 1).item())

        plt.imshow(mnist_test.test_data[r:r+1].view(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()