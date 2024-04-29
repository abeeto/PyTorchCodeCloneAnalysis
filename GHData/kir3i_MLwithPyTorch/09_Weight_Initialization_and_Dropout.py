'''
DNN(Deep Nueral Network)에서 발생하는 문제들과 해결책
1. weight initialization
    weight를 적절히 초기화해주는 것만으로도 학습 효율 향상을 꾀할 수 있다.
2. Dropout
    model이 training set에 Overfitting 되는 현상을 방지하기 위해 사용한다.
    일부 node를 확률적으로 제외시키는 방법으로, 속도 향상과 overfitting을 방지할 수 있다.
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

## set data
batch_size = 100
learning_rate = 0.001

mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

## set model
linear1 = nn.Linear(28*28, 256, bias=True)
linear2 = nn.Linear(256, 256, bias=True)
linear3 = nn.Linear(256, 10, bias=True)
relu = nn.ReLU()

# weight intialization ******
# linear1 = nn.Linear(28*28, 256, bias=True).to(device)
# linear2 = nn.Linear(256, 256, bias=True).to(device)
# linear3 = nn.Linear(256, 10, bias=True).to(device)
nn.init.xavier_uniform_(linear1.weight).to(device)  # Xavier Uniform initializatioin으로 초기화
nn.init.xavier_uniform_(linear2.weight).to(device)  # 08_ReLU와 같은 코드에 초기화만 바꿨더니 정확도가 98%로 올랐다!
nn.init.xavier_uniform_(linear3.weight).to(device)

# dropout
dropout = nn.Dropout(p=0.5)

#model = nn.Sequential(linear1, relu, linear2, relu, linear3)
model = nn.Sequential(linear1, relu, dropout,       # dropout 적용하여 model 선언
                      linear2, relu, dropout,       # 적용했더니 오히려 정확도는 97.8%로 소폭 감소
                      linear3).to(device)

## define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


## Learning
model.train()       # dropout 적용 시 train전에 호출해줘야 한다.
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
    print(f'epoch: {epoch+1:2d}/{nb_epochs} Cost: {avg_cost:.6f}')
print('Learning Completed')

with torch.no_grad():
    model.eval()    # dropout 적용 시 test하기 전에 호출해야 한다.
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