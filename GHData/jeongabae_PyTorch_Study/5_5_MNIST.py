#1. MNIST 데이터 이해하기
#MNIST는 숫자 0부터 9까지의 이미지로 구성된 손글씨 데이터셋
#MNIST 문제는 손글씨로 적힌 숫자 이미지가 들어오면, 그 이미지가 무슨 숫자인지 맞추는 문제

#784차원의 벡터로 만드는 코드를 미리보기로 보면
"""
for X, Y in data_loader:
  # 입력 이미지를 [batch_size × 784]의 크기로 reshape
  # 레이블은 원-핫 인코딩
  X = X.view(-1, 28*28)
"""

#2. 토치비전(torchvision) 소개하기
# torchvision은 유명한 데이터셋들, 이미 구현되어져 있는 유명한 모델들, 일반적인 이미지 전처리 도구들을 포함하고 있는 패키지
#torchvision에 어떤 데이터셋들(datasets)과 모델들(models) 그리고 어떤 전처리 방법들(transforms)을 제공

#3. 분류기 구현을 위한 사전 설정
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)
"""
구글의 Colab에서 '런타임 > 런타임 유형 변경 > 하드웨어 가속기 > GPU'를 선택하면 USE_CUDA의 값이 True가 되면서 '다음 기기로 학습합니다: 
cuda'라는 출력이 나옵니다. 즉, GPU로 연산하겠다는 의미입니다. 
반면에 '하드웨어 가속기 > None'을 선택하면 USE_CUDA의 값이 False가 되면서 '다음 기기로 학습합니다: cpu'라는 출력이 나옵니다. 
즉, CPU로 연산하겠다는 의미입니다.
"""

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# hyperparameters
training_epochs = 15
batch_size = 100

#4. MNIST 분류기 구현하기
#torchvision.datasets.dsets.MNIST를 사용하여 MNIST 데이터셋 불러오기
# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/', #첫번째 인자 root는 MNIST 데이터를 다운로드 받을 경로
                          train=True,       #. 두번째 인자 train은 인자로 True를 주면, MNIST의 훈련 데이터를 리턴받으며 False를 주면 테스트 데이터를 리턴받음.
                          transform=transforms.ToTensor(), #세번째 인자 transform은 현재 데이터를 파이토치 텐서로 변환
                          download=True) #네번째 인자 download는 해당 경로에 MNIST 데이터가 없다면 다운로드 받겠다는 의미

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
# dataset loader
#첫번째 인자인 dataset은 로드할 대상을 의미하며,
# 두번째 인자인 batch_size는 배치 크기,
# shuffle은 매 에포크마다 미니 배치를 셔플할 것인지의 여부,
# drop_last는 마지막 배치를 버릴 것인지
data_loader = DataLoader(dataset=mnist_train,
                                          batch_size=batch_size, # 배치 크기는 100
                                          shuffle=True,
                                          drop_last=True)

# MNIST data image of shape 28 * 28 = 784
linear = nn.Linear(784, 10, bias=True).to(device)

# 비용 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device) # 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # 배치 크기가 100이므로 아래의 연산에서 X는 (100, 784)의 텐서가 된다.
        X = X.view(-1, 28 * 28).to(device)
        # 레이블은 원-핫 인코딩이 된 상태가 아니라 0 ~ 9의 정수.
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
"""
Epoch: 0001 cost = 0.535468459
Epoch: 0002 cost = 0.359274209
Epoch: 0003 cost = 0.331187516
Epoch: 0004 cost = 0.316578060
Epoch: 0005 cost = 0.307158142
Epoch: 0006 cost = 0.300180763
Epoch: 0007 cost = 0.295130193
Epoch: 0008 cost = 0.290851474
Epoch: 0009 cost = 0.287417054
Epoch: 0010 cost = 0.284379572
Epoch: 0011 cost = 0.281825274
Epoch: 0012 cost = 0.279800713
Epoch: 0013 cost = 0.277808994
Epoch: 0014 cost = 0.276154339
Epoch: 0015 cost = 0.274440885
Learning finished
"""

# 테스트 데이터를 사용하여 모델을 테스트한다.
with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

    """
    Accuracy: 0.8883000016212463
    Label:  8
    Prediction:  3
    """