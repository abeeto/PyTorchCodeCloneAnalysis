#1. 데이터에 대한 이해(Data Definition)
# 예측을 위해 사용하는 데이터를 훈련 데이터셋(training dataset)
#학습이 끝난 후, 이 모델이 얼마나 잘 작동하는지 판별하는 데이터셋을 테스트 데이터셋(test dataset)

#1-2. 훈련 데이터셋의 구성
#데이터는 파이토치의 텐서의 형태(torch.tensor)
#_train은 공부한 시간, y_train은 그에 맵핑되는 점수
import torch

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

#2. 가설(Hypothesis) 수립 : 선형 회귀의 가설 : y=Wx+b
#W :가중치(Weight),  b : 편향(bias)

#3. 비용 함수(Cost function)에 대한 이해
#비용 함수(cost function) = 손실 함수(loss function) = 오차 함수(error function) = 목적 함수(objective function)
"""
오차들을 제곱해준 뒤에 전부 더하고 데이터의 개수인 으로 나누면, 오차의 제곱합에 대한 평균을 구할 수 있는데
이를 평균 제곱 오차(Mean Squared Error, MSE)라고 합니다.
"""
#비용함수 Cost(W,b)를 최소가 되게 만드는 W와 b를 구하면 훈련 데이터를 가장 잘 나타내는 직선을 구할 수 있습니다.

#4. 옵티마이저 - 경사 하강법(Gradient Descent)
"""
비용 함수(Cost Function)의 값을 최소로 하는 W와 b를 찾는 방법.
-> 이때 사용되는 것이 옵티마이저(Optimizer) 알고리즘 : 최적화 알고리즘이라고도 함.
가장 기본적인 옵티마이저 알고리즘인 경사 하강법(Gradient Descent)에 대해서 배움
"""
"""
 cost가 최소화가 되는 지점은 접선의 기울기가 0이 되는 지점이며, 또한 미분값이 0이 되는 지점입니다. 
 경사 하강법의 아이디어는 비용 함수(Cost function)를 미분하여 현재 W에서의 접선의 기울기를 구하고, 
 접선의 기울기가 낮은 방향으로 W의 값을 변경하는 작업을 반복하는 것에 있습니다.
"""
"""
기울기가 음수일 때 : W의 값이 증가
기울기가 양수일 때 : W의 값이 감소
"""
# 학습률(learning rate)이라고 말하는 α 는 어떤 의미를 가질까요? 학습률 α 은 의 W값을 변경할 때, 얼마나 크게 변경할지를 결정