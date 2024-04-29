# ImageFolder: 나만의 dataset으로 learning 수행하기
# 이번 코드는 참고용으로 작성한 것으로 실제 동작하지 않을 수 있다.
# 새로운 기법이 좀 나온다.
## 1. 학습한 모델을 저장하고 불러오는 방법(torch.save, model.load)
## 2. 학습할 img가 큰 경우 transform을 활용해 resize 하는 법
## 3. 멀티스레드로 데이터를 돌리는 방법 (DataLoader의 num_worker 매개변수)
## 4. test 시에도 DataLoader를 사용하기도 한다.
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# load dataset

trans = transforms.Compose([transforms.Resize(64, 128)])    # data transform function
train_data = torchvision.datasets.ImageFolder(root='custom_data/origin_data', transform=trans)   # img가 저장된 경로에서 data 불러옴

for num, value in enumerate(train_data):
    data, label = value
    if label == 0:
        data.save(f'custom_data/train_data/{num}_{label}.jpeg')    # transform을 적용한 채로 저장
    else:
        data.save(f'custom_data/train_data/{num}_{label}.jpeg')     # transform을 적용한 채로 저장

train_data = torchvision.datasets.ImageFolder(root='custom_data/train_data', transform=transforms.ToTensor())   # img가 저장된 경로에서 data 불러옴
data_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=2) # num_workers는 학습 시 사용할 subprocess(아마 스레드?)의 수를 결정한다.

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*13*29, 120),
            nn.ReLU(),
            nn.Linear(120, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

batches = len(data_loader)
nb_epochs = 3

for epoch in range(nb_epochs):
    avg_cost = 0
    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        
        # hypothesis
        hypothesis = model(X)

        # cost
        cost = F.cross_entropy(hypothesis, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost = cost / batches
    print(f'epoch: {epoch+1:2d}/{nb_epochs} Cost: {avg_cost:.6f}')
print('Learning Completed')

torch.save(model.state_dict(), './model/model.pth')   # 학습된 model을 저장한다!
new_model = CNN().to(device)
new_model.load_state_dict(torch.load('./model/model.pth'))  # 저장된 model을 load한다!


## test

# load test data
trans = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor()
])
test_data = torchvision.datasets.ImageFolder(root='./custom_data/test_data', transform=trans)

test_set = DataLoader(dataset=test_data, batch_size=len(test_data)) # batch 하나로 load함

# do test
with torch.no_grad():
    for X, Y in test_set:   # 어차피 batch 하나라서 한 번만 돌고 끝난다.
        X = X.to(device)
        Y = Y.to(device)

        hypothesis = model(X)
        isCorrect = torch.argmax(hypothesis, dim=-1) == Y
        accuracy = torch.mean(isCorrect.float())
        print(f'Accuracy: {accuracy*100:.3f}')