#다수의 x로부터 y를 예측하는 다중 선형 회귀(Multivariable Linear Regression)

#1. 데이터에 대한 이해(Data Definition)
"""
ex)3개의 퀴즈 점수로부터 최종 점수를 예측하는 모델
독립 변수 의 개수가 이제 1개가 아닌 3개일 때
H(x)=w1x1+w2x2+w3x3+b
"""

#2. 파이토치로 구현하기
#우선 필요한 도구들을 임포트하고 랜덤 시드를 고정
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

# 훈련 데이터 (x를 3개 선언)
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 가중치 w와 편향 b 초기화 (가중치 w도 3개 선언해주어야)
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#이제 가설, 비용 함수, 옵티마이저를 선언한 후에 경사 하강법을 1,000회 반복
# optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))
"""실행 결과
Epoch    0/1000 w1: 0.294 w2: 0.294 w3: 0.297 b: 0.003 Cost: 29661.800781
Epoch  100/1000 w1: 0.674 w2: 0.661 w3: 0.676 b: 0.008 Cost: 1.563634
Epoch  200/1000 w1: 0.679 w2: 0.655 w3: 0.677 b: 0.008 Cost: 1.497607
Epoch  300/1000 w1: 0.684 w2: 0.649 w3: 0.677 b: 0.008 Cost: 1.435026
Epoch  400/1000 w1: 0.689 w2: 0.643 w3: 0.678 b: 0.008 Cost: 1.375730
Epoch  500/1000 w1: 0.694 w2: 0.638 w3: 0.678 b: 0.009 Cost: 1.319511
Epoch  600/1000 w1: 0.699 w2: 0.633 w3: 0.679 b: 0.009 Cost: 1.266222
Epoch  700/1000 w1: 0.704 w2: 0.627 w3: 0.679 b: 0.009 Cost: 1.215696
Epoch  800/1000 w1: 0.709 w2: 0.622 w3: 0.679 b: 0.009 Cost: 1.167818
Epoch  900/1000 w1: 0.713 w2: 0.617 w3: 0.680 b: 0.009 Cost: 1.122429
Epoch 1000/1000 w1: 0.718 w2: 0.613 w3: 0.680 b: 0.009 Cost: 1.079378
"""
"""
위의 경우 가설을 선언하는 부분인 hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b에서도 
x_train의 개수만큼 w와 곱해주도록 작성해준 것을 확인할 수 있습니다.
"""

#3. 벡터와 행렬 연산으로 바꾸기
""" x_train1 ~ x_train1000을 전부 선언하고, w1 ~ w1000을 전부 선언해야 합니다.
다시 말해 와  변수 선언만 총 합 2,000개를 해야합니다.
또한 가설을 선언하는 부분에서도 마찬가지로 x_train과 w의 곱셈이 이루어지는 항을 1,000개를 작성해야 합니다. 이는 굉장히 비효율적입니다.
이를 해결하기 위해 행렬 곱셈 연산(또는 벡터의 내적)을 사용
"""
#행렬의 곱셈 과정에서 이루어지는 벡터 연산을 벡터의 내적(Dot Product)

#3-1. 벡터 연산으로 이해하기
#H(X)=w1x1+w2x2+w3x3  => H(X)=XW
#x의 개수가 3개였음에도 이제는 X와 W라는 두 개의 변수로 표현된 것을 볼 수 있습니다.

#3-2. 행렬 연산으로 이해하기
"""
전체 훈련 데이터의 개수를 셀 수 있는 1개의 단위를 샘플(sample)
각 샘플에서 y를 결정하게 하는 각각의 독립 변수 x를 특성(feature)
"""
# H(X)=XW+B
"""
결과적으로 전체 훈련 데이터의 가설 연산을 3개의 변수만으로 표현하였습니다.
이와 같이 벡터와 행렬 연산은 식을 간단하게 해줄 뿐만 아니라 다수의 샘플의 병렬 연산이므로 속도의 이점을 가집니다.

이를 참고로 파이토치로 구현
"""

#4. 행렬 연산을 고려하여 파이토치로 구현하기
x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  80],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
print(x_train.shape) #torch.Size([5, 3])
print(y_train.shape) #torch.Size([5, 1])

# 가중치와 편향 선언
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# X_train의 행렬의 크기는 (5 × 3)이며,  X벡터의 크기는 (3 × 1)이므로 두 행렬과 벡터는 행렬곱이 가능합니다.
# 행렬곱으로 가설을 선언
hypothesis = x_train.matmul(W) + b

#전체 코드!!!!!!!!!!!
x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  80],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))
    """
    Epoch    0/20 hypothesis: tensor([0., 0., 0., 0., 0.]) Cost: 29661.800781
Epoch    1/20 hypothesis: tensor([66.7178, 80.1701, 76.1025, 86.0194, 61.1565]) Cost: 9537.694336
Epoch    2/20 hypothesis: tensor([104.5421, 125.6208, 119.2478, 134.7862,  95.8280]) Cost: 3069.590088
Epoch    3/20 hypothesis: tensor([125.9858, 151.3882, 143.7087, 162.4333, 115.4844]) Cost: 990.670898
Epoch    4/20 hypothesis: tensor([138.1429, 165.9963, 157.5768, 178.1071, 126.6283]) Cost: 322.482086
Epoch    5/20 hypothesis: tensor([145.0350, 174.2780, 165.4395, 186.9928, 132.9461]) Cost: 107.717064
Epoch    6/20 hypothesis: tensor([148.9423, 178.9730, 169.8976, 192.0301, 136.5279]) Cost: 38.687496
Epoch    7/20 hypothesis: tensor([151.1574, 181.6346, 172.4254, 194.8856, 138.5585]) Cost: 16.499043
Epoch    8/20 hypothesis: tensor([152.4131, 183.1435, 173.8590, 196.5043, 139.7097]) Cost: 9.365656
Epoch    9/20 hypothesis: tensor([153.1250, 183.9988, 174.6723, 197.4217, 140.3625]) Cost: 7.071114
Epoch   10/20 hypothesis: tensor([153.5285, 184.4835, 175.1338, 197.9415, 140.7325]) Cost: 6.331847
Epoch   11/20 hypothesis: tensor([153.7572, 184.7582, 175.3958, 198.2360, 140.9424]) Cost: 6.092532
Epoch   12/20 hypothesis: tensor([153.8868, 184.9138, 175.5449, 198.4026, 141.0613]) Cost: 6.013817
Epoch   13/20 hypothesis: tensor([153.9602, 185.0019, 175.6299, 198.4969, 141.1288]) Cost: 5.986785
Epoch   14/20 hypothesis: tensor([154.0017, 185.0517, 175.6785, 198.5500, 141.1671]) Cost: 5.976325
Epoch   15/20 hypothesis: tensor([154.0252, 185.0798, 175.7065, 198.5800, 141.1888]) Cost: 5.971208
Epoch   16/20 hypothesis: tensor([154.0385, 185.0956, 175.7229, 198.5966, 141.2012]) Cost: 5.967835
Epoch   17/20 hypothesis: tensor([154.0459, 185.1045, 175.7326, 198.6059, 141.2082]) Cost: 5.964969
Epoch   18/20 hypothesis: tensor([154.0501, 185.1094, 175.7386, 198.6108, 141.2122]) Cost: 5.962291
Epoch   19/20 hypothesis: tensor([154.0524, 185.1120, 175.7424, 198.6134, 141.2145]) Cost: 5.959664
Epoch   20/20 hypothesis: tensor([154.0536, 185.1134, 175.7451, 198.6145, 141.2158]) Cost: 5.957089
    """