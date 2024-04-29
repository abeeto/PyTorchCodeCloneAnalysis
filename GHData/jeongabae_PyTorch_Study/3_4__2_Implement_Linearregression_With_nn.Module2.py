#2. 다중 선형 회귀 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

"""
데이터를 선언해줍니다. 여기서는 3개의 x로부터 하나의 y를 예측하는 문제입니다.
즉, 가설 수식은 H(x)=w1x1+w2x2+w3x3+b
"""
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

## 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1.
model = nn.Linear(3,1) # nn.Linear()는 입력의 차원, 출력의 차원을 인수로 받음.

print(list(model.parameters()))
"""첫번째 출력되는 것이 3개의 w고, 두번째 출력되는 것이 b에 해당(두 값 모두 현재는 랜덤 초기화가 되어져 있음)
# 두 출력 결과 모두 학습의 대상이므로 requires_grad=True가 되어져 있음
[Parameter containing:
tensor([[ 0.2975, -0.2548, -0.1119]], requires_grad=True), Parameter containing:
tensor([0.2710], requires_grad=True)]
"""

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
Epoch    0/2000 Cost: 31667.597656
... 중략 ...
Epoch 2000/2000 Cost: 0.199777
"""

#학습이 완료되었습니다. Cost의 값이 매우 작습니다.
# 3개의 w와 b의 값도 최적화가 되었는지 확인해봅시다.
#x에 임의의 입력 [73, 80, 75]를 넣어 모델이 예측하는 y의 값을 확인

# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]])
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var)

print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
#훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[151.2305]], grad_fn=<AddmmBackward>)
#당시 y의 값은 152였는데, 현재 예측값이 151이 나온 것으로 보아 어느정도는 3개의 w와 b의 값이 최적화 된것으로 보입니다.
# 이제 학습 후의 3개의 w와 b의 값을 출력

print(list(model.parameters()))
"""
[Parameter containing:
tensor([[0.9778, 0.4539, 0.5768]], requires_grad=True), Parameter containing:
tensor([0.2802], requires_grad=True)]
"""
