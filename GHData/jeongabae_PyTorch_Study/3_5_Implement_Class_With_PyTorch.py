#선형 회귀를 클래스로 구현
#1. 모델을 클래스로 구현하기
"""단순 선형 회귀 모델 구
import torch.nn as nn
# 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
model = nn.Linear(1,1)
"""

"""
#단순 선형 회귀 모델 클래스로 구현
import torch.nn as nn
class LinearRegressionModel(nn.Module): # torch.nn.Module을 상속받는 파이썬 클래스
def __init__(self): #모델의 구조와 동작을 정의하는 생성자를 정의
super().__init__() #super() 함수를 부르면 여기서 만든 클래스는 nn.Module 클래스의 속성들을 가지고 초기화
self.linear = nn.Linear(1, 1) # 단순 선형 회귀이므로 input_dim=1, output_dim=1.

def forward(self, x): # foward() 함수는 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 함수
#forward() 함수는 model 객체를 데이터와 함께 호출하면 자동으로 실행
#forward연산? H(x)식에 입력 x로부터 예측된 y를 얻는 것
return self.linear(x)

model = LinearRegressionModel()
"""

""" 다중 선형 회귀 모델 구현
import torch.nn as nn
# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1.
model = nn.Linear(3,1)
"""

"""
# 다중 선형 회귀 모델 클래스로 구현
class MultivariateLinearRegressionModel(nn.Module):
def __init__(self):
super().__init__()
self.linear = nn.Linear(3, 1) # 다중 선형 회귀이므로 input_dim=3, output_dim=1.

def forward(self, x):
return self.linear(x)

model = MultivariateLinearRegressionModel()
"""

#2. 단순 선형 회귀 클래스로 구현하기
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

class LinearRegressionModel(nn.Module):
def __init__(self):
super().__init__()
self.linear = nn.Linear(1, 1)

def forward(self, x):
return self.linear(x)

model = LinearRegressionModel()

# optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

# H(x) 계산
prediction = model(x_train)

# cost 계산
cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

# cost로 H(x) 개선하는 부분
# gradient를 0으로 초기화
optimizer.zero_grad()
# 비용 함수를 미분하여 gradient 계산
cost.backward() # backward 연산
# W와 b를 업데이트
optimizer.step()

if epoch % 100 == 0:
# 100번마다 로그 출력
print('Epoch {:4d}/{} Cost: {:.6f}'.format(
epoch, nb_epochs, cost.item()
))
"""

#3. 다중 선형 회귀 클래스로 구현하기
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

class MultivariateLinearRegressionModel(nn.Module):
def __init__(self):
super().__init__()
self.linear = nn.Linear(3, 1) # 다중 선형 회귀이므로 input_dim=3, output_dim=1.

def forward(self, x):
return self.linear(x)

model = MultivariateLinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs+1):

# H(x) 계산
prediction = model(x_train)
# model(x_train)은 model.forward(x_train)와 동일함.

# cost 계산
cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

# cost로 H(x) 개선하는 부분
# gradient를 0으로 초기화
optimizer.zero_grad()
# 비용 함수를 미분하여 gradient 계산
cost.backward()
# W와 b를 업데이트
optimizer.step()

if epoch % 100 == 0:
# 100번마다 로그 출력
print('Epoch {:4d}/{} Cost: {:.6f}'.format(
epoch, nb_epochs, cost.item()
))

"""