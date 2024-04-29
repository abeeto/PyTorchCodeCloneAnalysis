# 소프트맥스 회귀를 로우-레벨과 F.cross_entropy를 사용해서 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

#x_train의 각 샘플은 4개의 특성을 가지고 있으며, 총 8개의 샘플이 존재
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
#y_train은 각 샘플에 대한 레이블인데, 여기서는 0, 1, 2의 값을 가지는 것으로 보아 총 3개의 클래스가 존재
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

#1. 소프트맥스 회귀 구현하기(로우-레벨)
print(x_train.shape) #torch.Size([8, 4])
print(y_train.shape) #torch.Size([8])

#x_train의 크기는 8 × 4이며, y_train의 크기는 8 × 1입니다. 그런데 최종 사용할 레이블은 y_train에서 원-핫 인코딩을 한 결과이어야 합니다.
# 클래스의 개수는 3개이므로 y_train에 원-핫 인코딩한 결과는 8 × 3의 개수를 가져야 합니다.
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print(y_one_hot.shape) #torch.Size([8, 3])

#y_train에서 원-핫 인코딩을 한 결과인 y_one_hot의 크기는 8 × 3입니다. 즉, W 행렬의 크기는 4 × 3이어야
# 모델 초기화 : W와 b를 선언
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정 : 옵티마이저로는 경사 하강법을 사용
optimizer = optim.SGD([W, b], lr=0.1)

#F.softmax()와 torch.log()를 사용하여 가설과 비용 함수를 정의하고, 총 1,000번의 에포크를 수행
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # 가설
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)

    # 비용 함수
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

#2. 소프트맥스 회귀 구현하기(하이-레벨)
#->F.cross_entropy()는 그 자체로 소프트맥스 함수를 포함하고 있으므로 가설에서는 소프트맥스 함수를 사용할 필요가 없습니다.
# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

#3. 소프트맥스 회귀 nn.Module로 구현하기
#->선형 회귀에서 구현에 사용했던 nn.Linear()를 사용합니다.
# ->output_dim이 1이었던 선형 회귀때와 달리 output_dim은 이제 클래스의 개수여야 합니다.
# 모델을 선언 및 초기화. 4개의 특성을 가지고 3개의 클래스로 분류. input_dim=4, output_dim=3.
model = nn.Linear(4, 3)

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

# F.cross_entropy()를 사용할 것이므로 따로 소프트맥스 함수를 가설에 정의하지 않습니다.
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

#4. 소프트맥스 회귀 클래스로 구현하기
#->소프트맥스 회귀를 nn.Module을 상속받은 클래스로 구현
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Output이 3!

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))