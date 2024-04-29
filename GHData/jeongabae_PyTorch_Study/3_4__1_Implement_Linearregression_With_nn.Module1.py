#파이토치에서 이미 구현되어져 제공되고 있는 함수들을 불러오는 것으로 더 쉽게 선형 회귀 모델을 구현
#ex)파이토치에서는 선형 회귀 모델이 nn.Linear()라는 함수로, 또 평균 제곱오차가 nn.functional.mse_loss()라는 함수로 구현되어져 있음.

"""두 함수의 사용 예제
import torch.nn as nn
model = nn.Linear(input_dim, output_dim)

import torch.nn.functional as F
cost = F.mse_loss(prediction, y_train)
"""

#1. 단순 선형 회귀 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터를 선언합니다. 아래 데이터는 y=2x를 가정된 상태에서 만들어진 데이터로
# 우리는 이미 정답이 W=2, b=0임을 알고 있는 사태입니다.
# 모델이 이 두 W와 b의 값을 제대로 찾아내도록 하는 것이 목표
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
model = nn.Linear(1,1) #nn.Linear()는 입력의 차원, 출력의 차원을 인수로 받음.

#model에는 가중치 W와 편향 b가 저장되어져 있습니다. 이 값은 model.parameters()라는 함수를 사용하여 불러올 수 있음.
print(list(model.parameters()))
""" 첫번째 값이 W고, 두번째 값이 b에 해당, 두 값 모두 학습의 대상이므로 requires_grad=True
[Parameter containing:
tensor([[0.5153]], requires_grad=True), Parameter containing:
tensor([-0.4414], requires_grad=True)]
"""

# optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # model.parameters()를 사용하여 W와 b를 전달

# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train) #prediction = model(x_train)은 x_train으로부터 예측값을 리턴하므로 forward 연산

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산 : 학습 과정에서 비용 함수를 미분하여 기울기를 구하는 것
                    #cost.backward()는 비용 함수로부터 기울기를 구하라는 의미이며 backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
"""학습이 완료됨. Cost의 값이 매우 작다.
Epoch    0/2000 Cost: 13.103540
... 중략 ...
Epoch 2000/2000 Cost: 0.000000
"""

#아래는 x에 임의의 값 4를 넣어 모델이 예측하는 y의 값을 확인해보겠습니다.
# 임의의 입력 4를 선언
new_var =  torch.FloatTensor([[4.0]])
# 입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) # forward 연산 :  H(x)식에 입력 x로부터 예측된 y를 얻는 것
                        #pred_y = model(new_var)는 임의의 값 new_var로부터 예측값을 리턴하므로 forward 연산

# y = 2x 이므로 입력이 4라면 y가 8에 가까운 값이 나와야 제대로 학습이 된 것
print("훈련 후 입력이 4일 때의 예측값 :", pred_y)
"""
훈련 후 입력이 4일 때의 예측값 : tensor([[7.9989]], grad_fn=<AddmmBackward>)
#이 문제의 정답은 y=2x가 정답이므로 y값이 8에 가까우면 W와 b의 값이 어느정도 최적화가 된 것으로 볼 수 있습니다. 
#실제로 예측된 y값은 7.9989로 8에 매우 가깝습니다.
"""

# 학습 후의 W와 b의 값을 출력
print(list(model.parameters()))
"""
#W의 값이 2에 가깝고, b의 값이 0에 가까운 것을 볼 수 있다.
[Parameter containing:
tensor([[1.9994]], requires_grad=True), Parameter containing:
tensor([0.0014], requires_grad=True)]
"""