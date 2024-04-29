#둘 중 하나를 결정하는 문제를 이진 분류(Binary Classification)
# 그리고 이진 분류를 풀기 위한 대표적인 알고리즘으로 로지스틱 회귀(Logistic Regression)가 있음
#->알고리즘의 이름은 회귀이지만 실제로는 분류(Classification) 작업에 사용

#1. 이진 분류(Binary Classification)
#로지스틱 회귀의 가설은 선형 회귀 때의 H(x)=Wx+b가 아니라,
# S자 모양의 그래프를 만들 수 있는 어떤 특정 함수 f를 추가적으로 사용하여 H(x)=f(Wx+b)의 가설을 사용
#S자 모양의 그래프를 그릴 수 있는 어떤 함수 f가 이미 널리 알려져있음. 바로 시그모이드 함수.

#2. 시그모이드 함수(Sigmoid function)
#선형 회귀에서는 최적의 W와 h를 찾는 것이 목표. 여기서도 그게 목표.
#파이썬에서는 그래프를 그릴 수 있는 도구로서 Matplotlib을 사용할 수 있습니다.
"""
%matplotlib inline #아나콘다(Anaconda)를 통해 Juptyer Notebook에서 사용되는 코드
                  #파이참에서 %matplolib inline를 사용하고 싶다면 plt.show()를 대신 작성하여 사용
"""

import numpy as np # 넘파이 사용
import matplotlib.pyplot as plt # 맷플롯립사용


def sigmoid(x): # 시그모이드 함수 정의
    return 1/(1+np.exp(-x))

#1. W가 1이고 b가 0인 그래프
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
"""
위의 그래프를 통해시그모이드 함수는 출력값을 0과 1사이의 값으로 조정하여 반환함을 알 수 있습니다. 
x가 0일 때 0.5의 값을 가집니다. x가 매우 커지면 1에 수렴합니다. 반면, x가 매우 작아지면 0에 수렴합니다.
"""

#2. W값의 변화에 따른 경사도의 변화
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--') # W의 값이 0.5일때 빨간색선
plt.plot(x, y2, 'g') # W의 값이 1일때  초록색선
plt.plot(x, y3, 'b', linestyle='--') # W의 값이 2일때 파란색선
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show() #위의 그래프는 b의 값에 따라서 그래프가 좌, 우로 이동하는 것을 보여줍니다.

#4. 시그모이드 함수를 이용한 분류
"""
시그모이드 함수는 입력값이 한없이 커지면 1에 수렴하고, 입력값이 한없이 작아지면 0에 수렴합니다.
시그모이드 함수의 출력값은 0과 1 사이의 값을 가지는데 이 특성을 이용하여 분류 작업에 사용할 수 있습니다.
예를 들어 임계값을 0.5라고 정해보겠습니다. 출력값이 0.5 이상이면 1(True), 0.5이하면 0(False)으로 판단하도록 할 수 있습니다. 
이를 확률이라고 생각하면 해당 레이블에 속할 확률이 50%가 넘으면 해당 레이블로 판단하고,
 해당 레이블에 속할 확률이 50%보다 낮으면 아니라고 판단하는 것으로 볼 수 있습니다.
"""

#3. 비용 함수(Cost function)
#비용 함수 수식에서 가설은 이제 H(x)=Wx+b가 아니라 H(x)=sigmoid(Wx+b)입니다.
"""
시그모이드 함수의 특징은 함수의 출력값이 0과 1사이의 값이라는 점입니다.
즉, 실제값이 1일 때 예측값이 0에 가까워지면 오차가 커져야 하며, 실제값이 0일 때, 예측값이 1에 가까워지면 오차가 커져야 합니다. 
그리고 이를 충족하는 함수가 바로 로그 함수입니다.
다음은 y=0.5에 대칭하는 두 개의 로그 함수 그래프입니다.
"""

#4. 파이토치로 로지스틱 회귀 구현하기
#파이토치로 로지스틱 회귀 중에서도 다수의 x로 부터 y를 예측하는 다중 로지스틱 회귀를 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

#x_train과 y_train을 텐서로 선언
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape) #torch.Size([6, 2])
print(y_train.shape) #torch.Size([6, 1])

"""
 x_train을 X라고 하고, 이와 곱해지는 가중치 벡터를 W라고 하였을 때, 
 XW가 성립되기 위해서는  W벡터의 크기는 2 × 1이어야 
"""
#W와 b를 선언
W = torch.zeros((2, 1), requires_grad=True) # 크기는 2 x 1
b = torch.zeros(1, requires_grad=True)

#가설식을 세워보겠습니다. 파이토치에서는 e^x를 구현하기 위해서 torch.exp(x)를 사용
hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b))) #행렬 연산을 사용한 가설식

# W와 b는 torch.zeros를 통해 전부 0으로 초기화 된 상태입니다. 이 상태에서 예측값을 출력
print(hypothesis) # 예측값인 H(x) 출력
"""실제값 y_train과 크기가 동일한 6 × 1의 크기를 가지는 예측값 벡터가 나오는데 모든 값이 0.5
tensor([[0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000]], grad_fn=<MulBackward>)
"""

#다음은 torch.sigmoid를 사용하여 좀 더 간단히 구현한 가설식(더 간단)
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)
"""
tensor([[0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000]], grad_fn=<SigmoidBackward>)
"""

print(hypothesis)# 현재 예측값
print(y_train)# 실제값
"""
tensor([[0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000]], grad_fn=<SigmoidBackward>)
tensor([[0.],
        [0.],
        [0.],
        [1.],
        [1.],
        [1.]])
"""
"""
#현재 총 6개의 원소가 존재하지만 하나의 샘플. 즉, 하나의 원소에 대해서만 오차를 구하는 식을 작성해보자
-(y_train[0] * torch.log(hypothesis[0]) +
  (1 - y_train[0]) * torch.log(1 - hypothesis[0]))
#tensor([0.6931], grad_fn=<NegBackward>)
"""
losses = -(y_train * torch.log(hypothesis) +
           (1 - y_train) * torch.log(1 - hypothesis))
print(losses)
"""
tensor([[0.6931],
        [0.6931],
        [0.6931],
        [0.6931],
        [0.6931],
        [0.6931]], grad_fn=<NegBackward>)
"""

#전체 오차에 대한 평균
cost = losses.mean()
print(cost)#tensor(0.6931, grad_fn=<MeanBackward1>)
#결과적으로 얻은 cost는 0.6931

"""
#지금까지 비용 함수의 값을 직접 구현하였는데, 사실 파이토치에서는 로지스틱 회귀의 비용 함수를 이미 구현해서 제공
#사용 방법은 torch.nn.functional as F와 같이 임포트 한 후에 F.binary_cross_entropy(예측값, 실제값)과 같이 사용하면됨
F.binary_cross_entropy(hypothesis, y_train)
"""

#모델의 훈련 과정까지 추가한 전체 코드
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
# 모델 초기화
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = -(y_train * torch.log(hypothesis) +
             (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
"""
Epoch    0/1000 Cost: 0.693147
... 중략 ...
Epoch 1000/1000 Cost: 0.019852
"""

"""
학습이 끝났습니다. 이제 훈련했던 훈련 데이터를 그대로 입력으로 사용했을 때, 제대로 예측하는지 확인해보겠습니다.
현재 W와 b는 훈련 후의 값을 가지고 있습니다. 현재 W와 b를 가지고 예측값을 출력해보겠습니다.
"""
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)
"""
tensor([[2.7648e-04],
        [3.1608e-02],
        [3.8977e-02],
        [9.5622e-01],
        [9.9823e-01],
        [9.9969e-01]], grad_fn=<SigmoidBackward>)
"""

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
"""실제값은 [[0], [0], [0], [1], [1], [1]]이므로, 이는 결과적으로 False, False, False, True, True, True와 동일
tensor([[False],
        [False],
        [False],
        [ True],
        [ True],
        [ True]])
"""
print(W)
print(b)
"""훈련이 된 후의 W와 b의 값을 출력
tensor([[3.2530],
        [1.5179]], requires_grad=True)
tensor([-14.4819], requires_grad=True)
"""