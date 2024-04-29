import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 현재 실습하고 있는 파이썬 코드를 재실행해도 다음에도 같은 결과가 나오도록 랜덤 시드(random seed)를 줍니다.
torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

print(x_train)
print(x_train.shape)
"""
tensor([[1.],
        [2.],
        [3.]])
torch.Size([3, 1])
"""

print(y_train)
print(y_train.shape)
"""
tensor([[2.],
        [4.],
        [6.]])
torch.Size([3, 1])
"""

#3. 가중치와 편향의 초기화
"""
선형 회귀란 학습 데이터와 가장 잘 맞는 하나의 직선을 찾는 일입니다.
그리고 가장 잘 맞는 직선을 정의하는 것은 바로 W와 b입니다.
선형 회귀의 목표는 가장 잘 맞는 직선을 정의하는 W와 b의 값을 찾는 것입니다.
"""
# 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함.
W = torch.zeros(1, requires_grad=True)
# 가중치 W를 출력
print(W)#tensor([0.], requires_grad=True)
"""
가중치 W가 0으로 초기화되어있으므로 0이 출력된 것을 확인할 수 있습니다.
 위에서 requires_grad=True가 인자로 주어진 것을 확인할 수 있습니다. 이는 이 변수는 학습을 통해 계속 값이 변경되는 변수
"""

#마찬가지로 편향 b도 0으로 초기화하고, 학습을 통해 값이 변경되는 변수
b = torch.zeros(1, requires_grad=True)
print(b) #tensor([0.], requires_grad=True)

#현재 가중치 W와 b 둘 다 0이므로 현 직선의 방정식은 y=0*x+0
#지금 상태에선 x에 어떤 값이 들어가도 가설은 0을 예측하게 됩니다. 즉, 아직 적절한 와 의 값이 아닙니다.

#4. 가설 세우기
#파이토치 코드 상으로 직선의 방정식에 해당되는 가설을 선언 H(x)=Wx+b
hypothesis = x_train * W + b
print(hypothesis)
"""
tensor([[0.],
        [0.],
        [0.]], grad_fn=<AddBackward0>)
"""
#5. 비용 함수 선언하기
# 앞서 배운 torch.mean으로 평균을 구한다.
cost = torch.mean((hypothesis - y_train) ** 2)
print(cost) #tensor(18.6667, grad_fn=<MeanBackward1>)

#6. 경사 하강법 구현하기
"""
이제 경사 하강법을 구현합니다. 아래의 'SGD'는 경사 하강법의 일종입니다. lr은 학습률(learning rate)를 의미합니다.
학습 대상인 W와 b가 SGD의 입력이 됩니다.
"""
optimizer = optim.SGD([W, b], lr=0.01)
"""
optimizer.zero_grad()를 실행하므로서 미분을 통해 얻은 기울기를 0으로 초기화합니다. 
기울기를 초기화해야만 새로운 가중치 편향에 대해서 새로운 기울기를 구할 수 있습니다. 
그 다음 cost.backward() 함수를 호출하면 가중치 W와 편향 b에 대한 기울기가 계산됩니다. 
그 다음 경사 하강법 최적화 함수 opimizer의 .step() 함수를 호출하여 
인수로 들어갔던 W와 b에서 리턴되는 변수들의 기울기에 학습률(learining rate) 0.01을 곱하여 빼줌으로서 업데이트합니다.
"""
# gradient를 0으로 초기화
optimizer.zero_grad()
# 비용 함수를 미분하여 gradient 계산
cost.backward()
# W와 b를 업데이트
optimizer.step()

#7. 전체 코드
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1999 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
""" 
결과적으로 훈련 과정에서 W와 b는 훈련 데이터와 잘 맞는 직선을 표현하기 위한 적절한 값으로 변화해갑니다. 
에포크(Epoch)는 전체 훈련 데이터가 학습에 한 번 사용된 주기를 말합니다.이번 실습의 경우 2,000번을 수행
Epoch    0/2000 W: 0.187, b: 0.080 Cost: 18.666666
Epoch  100/2000 W: 1.746, b: 0.578 Cost: 0.048171
Epoch  200/2000 W: 1.800, b: 0.454 Cost: 0.029767
Epoch  300/2000 W: 1.843, b: 0.357 Cost: 0.018394
Epoch  400/2000 W: 1.876, b: 0.281 Cost: 0.011366
Epoch  500/2000 W: 1.903, b: 0.221 Cost: 0.007024
Epoch  600/2000 W: 1.924, b: 0.174 Cost: 0.004340
Epoch  700/2000 W: 1.940, b: 0.136 Cost: 0.002682
Epoch  800/2000 W: 1.953, b: 0.107 Cost: 0.001657
Epoch  900/2000 W: 1.963, b: 0.084 Cost: 0.001024
Epoch 1000/2000 W: 1.971, b: 0.066 Cost: 0.000633
Epoch 1100/2000 W: 1.977, b: 0.052 Cost: 0.000391
Epoch 1200/2000 W: 1.982, b: 0.041 Cost: 0.000242
Epoch 1300/2000 W: 1.986, b: 0.032 Cost: 0.000149
Epoch 1400/2000 W: 1.989, b: 0.025 Cost: 0.000092
Epoch 1500/2000 W: 1.991, b: 0.020 Cost: 0.000057
Epoch 1600/2000 W: 1.993, b: 0.016 Cost: 0.000035
Epoch 1700/2000 W: 1.995, b: 0.012 Cost: 0.000022
Epoch 1800/2000 W: 1.996, b: 0.010 Cost: 0.000013
Epoch 1900/2000 W: 1.997, b: 0.008 Cost: 0.000008
Epoch 2000/2000 W: 1.997, b: 0.006 Cost: 0.000005
최종 훈련 결과를 보면 최적의 기울기 W는 2에 가깝고, b는 0에 가까운 것을 볼 수 있습니다.
현재 훈련 데이터가 x_train은 [[1], [2], [3]]이고 y_train은 [[2], [4], [6]]인 것을 감안하면
실제 정답은 W가 2이고, b가 0인 H(x)=2x이므로 거의 정답을 찾은 셈입니다.
    """

#5. optimizer.zero_grad()가 필요한 이유
#파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징이 있음
#계속해서 미분값인 2가 누적되는 것을 볼 수 있습니다. 그렇기 때문에 optimizer.zero_grad()를 통해 미분값을 계속 0으로 초기화시켜줘야함
"""
import torch
w = torch.tensor(2.0, requires_grad=True)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

  z = 2*w

  z.backward()
  print('수식을 w로 미분한 값 : {}'.format(w.grad))
"""
""" 실행결과
수식을 w로 미분한 값 : 2.0
수식을 w로 미분한 값 : 4.0
수식을 w로 미분한 값 : 6.0
수식을 w로 미분한 값 : 8.0
수식을 w로 미분한 값 : 10.0
수식을 w로 미분한 값 : 12.0
수식을 w로 미분한 값 : 14.0
수식을 w로 미분한 값 : 16.0
수식을 w로 미분한 값 : 18.0
수식을 w로 미분한 값 : 20.0
수식을 w로 미분한 값 : 22.0
수식을 w로 미분한 값 : 24.0
수식을 w로 미분한 값 : 26.0
수식을 w로 미분한 값 : 28.0
수식을 w로 미분한 값 : 30.0
수식을 w로 미분한 값 : 32.0
수식을 w로 미분한 값 : 34.0
수식을 w로 미분한 값 : 36.0
수식을 w로 미분한 값 : 38.0
수식을 w로 미분한 값 : 40.0
수식을 w로 미분한 값 : 42.0
"""

#6. torch.manual_seed()를 하는 이유
"""
torch.manual_seed()를 사용한 프로그램의 결과는 다른 컴퓨터에서 실행시켜도 동일한 결과를 얻을 수 있습니다. 
그 이유는 torch.manual_seed()는 난수 발생 순서와 값을 동일하게 보장해준다는 특징때문입니다. 
"""
"""
import torch
torch.manual_seed(3)
print('랜덤 시드가 3일 때')
for i in range(1,3):
  print(torch.rand(1))
"""
"""실행결과
랜덤 시드가 3일 때
tensor([0.0043])
tensor([0.1056])
"""

"""
torch.manual_seed(5)
print('랜덤 시드가 5일 때')
for i in range(1,3):
  print(torch.rand(1))
"""
"""실행결과
랜덤 시드가 5일 때
tensor([0.8303])
tensor([0.1261])
"""

"""
#0.8303과 0.1261이 나옵니다. 
#이제 다시 랜덤 시드값을 3으로 돌려보겠습니다. 이렇게 하면 프로그램을 다시 처음부터 실행한 것처럼 난수 발생 순서가 초기화됩니다.
import torch
torch.manual_seed(3)
print('랜덤 시드가 3일 때')
for i in range(1,3):
  print(torch.rand(1))
"""
"""실행결과 : 다시 동일하게 0.0043과 0.1056이 나옵니다
랜덤 시드가 3일 때
tensor([0.0043])
tensor([0.1056])
"""

#cf텐서에는 requires_grad라는 속성이 있습니다. 이것을 True로 설정하면 자동 미분 기능이 적용됩니다.
# 선형 회귀부터 신경망과 같은 복잡한 구조에서 파라미터들이 모두 이 기능이 적용됩니다.
# requires_grad = True가 적용된 텐서에 연산을 하면, 계산 그래프가 생성되며 backward 함수를 호출하면 그래프로부터 자동으로 미분이 계산됩니다.