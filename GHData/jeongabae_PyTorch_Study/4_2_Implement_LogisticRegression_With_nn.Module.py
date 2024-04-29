#비용 함수 수식에서 가설은 이제 H(x)=Wx+b가 아니라 H(x)=sigmoid(Wx+b)입니다.
#파이토치에서는 nn.Sigmoid()를 통해서 시그모이드 함수를 구현하므로
# 결과적으로 nn.Linear()의 결과를 nn.Sigmoid()를 거치게하면 로지스틱 회귀의 가설식이 됩니다.

#1. 파이토치의 nn.Linear와 nn.Sigmoid로 로지스틱 회귀 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

#훈련 데이터를 텐서로 선언
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

#nn.Sequential()은 nn.Module 층을 차례로 쌓을 수 있도록함
#nn.Sequential()은 Wx+b와 같은 수식과 시그모이드 함수 등과 같은 여러 함수들을 연결해주는 역할
model = nn.Sequential(
   nn.Linear(2, 1), # input_dim = 2, output_dim = 1
   nn.Sigmoid() # 출력은 시그모이드 함수를 거친다
)

#현재 W와 b는 랜덤 초기화가 된 상태입니다. 훈련 데이터를 넣어 예측값을 확인
model(x_train)
"""현재 W와 b는 임의의 값을 가지므로 현재의 예측은 의미가 없습니다.
tensor([[0.4020],
        [0.4147],
        [0.6556],
        [0.5948],
        [0.6788],
        [0.8061]], grad_fn=<SigmoidBackward>)
"""

# 경사 하강법을 사용하여 훈련. 총 100번의 에포크를 수행
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))
"""중간부터 정확도는 100%가 나오기 시작
Epoch    0/1000 Cost: 0.539713 Accuracy 83.33%
... 중략 ...
Epoch 1000/1000 Cost: 0.019843 Accuracy 100.00%
"""

model(x_train)
"""
tensor([[0.0240],
        [0.1476],
        [0.2739],
        [0.7967],
        [0.9491],
        [0.9836]], grad_fn=<SigmoidBackward>)
"""
"""
0.5를 넘으면 True, 그보다 낮으면 False로 간주합니다. 실제값은 [[0], [0], [0], [1], [1], [1]]입니다.
 이는 False, False, False, True, True, True에 해당되므로 전부 실제값과 일치하도록 예측한 것을 확인할 수 있습니다.
"""

#훈련 후의 W와 b의 값을 출력
print(list(model.parameters()))
"""
print(list(model.parameters()))
[Parameter containing:
tensor([[3.2534, 1.5181]], requires_grad=True), Parameter containing:
tensor([-14.4839], requires_grad=True)]
"""

#2. 인공 신경망으로 표현되는 로지스틱 회귀.