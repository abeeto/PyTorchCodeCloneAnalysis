#1. 미니 배치와 배치 크기(Mini Batch and Batch Size)
# 전체 데이터를 더 작은 단위로 나누어서 해당 단위로 학습하는 개념이 나오게 되었습니다. => 이 단위를 미니 배치(Mini Batch)
"""
 미니 배치 학습을 하게되면 미니 배치만큼만 가져가서 미니 배치에 대한 대한 비용(cost)를 계산하고, 경사 하강법을 수행합니다.
 그리고 다음 미니 배치를 가져가서 경사 하강법을 수행하고 마지막 미니 배치까지 이를 반복합니다.
 이렇게 전체 데이터에 대한 학습이 1회 끝나면 1 에포크(Epoch)가 끝나게 됩니다.
 cf)에포크(Epoch) : 전체 훈련 데이터가 학습에 한 번 사용된 주기

 전체 데이터에 대해서 한 번에 경사 하강법을 수행하는 방법을 '배치 경사 하강법'
  반면, 미니 배치 단위로 경사 하강법을 수행하는 방법을 '미니 배치 경사 하강법'

배치 경사 하강법은 경사 하강법을 할 때,
전체 데이터를 사용하므로 가중치 값이 최적값에 수렴하는 과정이 매우 안정적이지만, 계산량이 너무 많이 듭니다.
미니 배치 경사 하강법은 경사 하강법을 할 때,
전체 데이터의 일부만을 보고 수행하므로 최적값으로 수렴하는 과정에서 값이 조금 헤매기도 하지만 훈련 속도가 빠릅니다.

배치 크기는 보통 2의 제곱수를 사용
"""

#2. 이터레이션(Iteration) : 한 번의 에포크 내에서 이루어지는 매개변수인 가중치 W와 b의 업데이트 횟수
# 전체 데이터가 2,000일 때 배치 크기를 200으로 한다면 이터레이션의 수는 총 10개입니다. 이는 한 번의 에포크 당 매개변수 업데이트가 10번 이루어짐을 의미

#3. 데이터 로드하기(Data Load)
#파이토치에서는 데이터를 좀 더 쉽게 다룰 수 있도록 유용한 도구로서 데이터셋(Dataset)과 데이터로더(DataLoader)를 제공
#이를 사용하면 미니 배치 학습, 데이터 셔플(shuffle), 병렬 처리까지 간단히 수행 가능
#기본적인 사용 방법은 Dataset을 정의하고, 이를 DataLoader에 전달하는 것

#텐서를 입력받아 Dataset의 형태로 변환해주는 TensorDataset을 사용해보자
import torch
import torch.nn as nn
import torch.nn.functional as F

#TensorDataset과 DataLoader를 임포트합니다.
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

#TensorDataset은 기본적으로 텐서를 입력으로 받습니다. 텐서 형태로 데이터를 정의
x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  90],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

#이제 이를 TensorDataset의 입력으로 사용하고 dataset으로 저장
dataset = TensorDataset(x_train, y_train)

"""
파이토치의 데이터셋을 만들었다면 데이터로더를 사용 가능합니다. 
데이터로더는 기본적으로 2개의 인자를 입력받는다. 하나는 데이터셋, 미니 배치의 크기입니다. 
이때 미니 배치의 크기는 통상적으로 2의 배수를 사용합니다. (ex) 64, 128, 256...)
 그리고 추가적으로 많이 사용되는 인자로 shuffle이 있습니다. 
shuffle=True를 선택하면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꿉니다.
(순서에 익숙해지는 것을 방지하여 학습할 때는 이 옵션을 True로 주는 걸 권장)
"""
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#모델과 옵티마이저를 설계
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

#훈련을 진행
nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    # print(batch_idx)
    # print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
"""Cost의 값이 점차 작아집니다. (에포크를 더 늘려서 훈련하면 Cost의 값이 더 작아질 수 있음)
Epoch    0/20 Batch 1/3 Cost: 26085.919922
Epoch    0/20 Batch 2/3 Cost: 3660.022949
Epoch    0/20 Batch 3/3 Cost: 2922.390869
... 중략 ...
Epoch   20/20 Batch 1/3 Cost: 6.315856
Epoch   20/20 Batch 2/3 Cost: 13.519956
Epoch   20/20 Batch 3/3 Cost: 4.262849
"""

#모델의 입력으로 임의의 값을 넣어 예측값을 확인
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]])
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
"""
훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[154.3850]], grad_fn=<AddmmBackward>)
"""