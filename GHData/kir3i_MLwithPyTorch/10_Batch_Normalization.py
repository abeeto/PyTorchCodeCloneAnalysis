'''
여러 Layer로 구성된 model을 이용하여 ML을 수행할 시 발생하는 문제점
  - Gradient Vanishing
  - Gradient Exploding

기존에는 위의 문제를 해결하기 위해 다음과 같은 방법을 사용함
  - Change activation function: activation function을 변경
  - Careful initialization: 초기화를 잘 하기
  - Small learning rate: lr을 줄여서 해결

그러나 위의 방법은 간접적인 방법이다.
2015년에 논문으로 새로 발표된 직접적인 방법
  - Batch normalization
현재까지 제일 효과적인 해결법이라고 한다.
위 해결법에서는 Gradient Vanishing 문제와 Exploding문제의 원인을 
Internal Covariate Shift때문이라고 설명하고 (분포 쏠림 현상)
ICS 현상의 해결책으로 batch normalization을 제시한다.
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

# set parameter
batch_size = 100
learning_rate = 0.001

# load data
mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

# set model
linear1 = nn.Linear(28*28, 256, bias=True)
linear2 = nn.Linear(256, 256, bias=True)
linear3 = nn.Linear(256, 10, bias=True)
relu = nn.ReLU()
bn1 = nn.BatchNorm1d(256)   # batch normalization
bn2 = nn.BatchNorm1d(256)   # batch normalization
# 1. 비교를 위해 특별한 초기화 없이 테스트했더니 정확도가 80% 조금 안되게 나왔다.
# 2. 초기화 하면? 83.57%의 정확도를 보인다. 생각보다 별로 안 높네??
nn.init.xavier_uniform_(linear1.weight).to(device)
nn.init.xavier_uniform_(linear2.weight).to(device)
nn.init.xavier_uniform_(linear3.weight).to(device)

dropout = nn.Dropout(p=0.5) # 3. dropout 적용해봤다. 96.41%의 정확도를 보인다. 성공적
model = nn.Sequential(linear1, bn1, relu, dropout,
                      linear2, bn2, relu, dropout,
                      linear3).to(device)

# set optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# learning
nb_epochs = 15
model.train()   # batch norm 적용 시 train전에 호출해줘야 한다.

for epoch in range(nb_epochs):
    avg_cost = 0
    batches = len(data_loader)
    for X, Y in data_loader:
        X = X.view(-1, 28*28).to(device)    # 적절한 view로 바꿈
        # hypothesis
        hypothesis = model(X)
        # cost
        cost = F.cross_entropy(hypothesis, Y)   # Softmax니까 cross_entropy로 cost 계산
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / batches
    print(f'epoch: {epoch+1:2d}/{nb_epochs} Cost: {avg_cost:.6f}')
print('Learning Finished')

# evaluation
with torch.no_grad():
    model.eval()    # batch norm 적용 시 test전에 호출해줘야 한다.
    
    # set data
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    # hypothesis
    hypothesis = model(X_test)
    # accuracy
    isCorrect = torch.argmax(hypothesis, dim=-1) == Y_test
    accuracy = torch.mean(isCorrect.float())
    print(f'Accuracy: {accuracy*100:.3f}%')

    # visualization test
    for _ in range(10):
        r = random.randint(0, len(mnist_test)-1)
        X_single_data = mnist_test.test_data[r:r+1].view(-1, 28*28).float().to(device)
        Y_single_data = mnist_test.test_labels[r:r+1].to(device)

        print('Label: ', Y_single_data.item())
        single_hypothesis = model(X_single_data)
        print('Prediction: ', torch.argmax(single_hypothesis, dim=-1).item())

        plt.imshow(mnist_test.test_data[r:r+1].view(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()