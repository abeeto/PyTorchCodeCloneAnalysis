#기본적인 사용 방법은 Dataset을 정의하고, 이를 DataLoader에 전달하는 것
# torch.utils.data.Dataset을 상속받아 직접 커스텀 데이터셋(Custom Dataset)을 만드는 경우도 있음!
#torch.utils.data.Dataset은 파이토치에서 데이터셋을 제공하는 추상 클래스입니다.
# Dataset을 상속받아 다음 메소드들을 오버라이드 하여 커스텀 데이터셋 만들기
"""커스텀 데이터셋을 만들 때, 가장 기본적인 뼈대
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self):
  데이터셋의 전처리를 해주는 부분

  def __len__(self):
  데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분

  def __getitem__(self, idx):
  데이터셋에서 특정 1개의 샘플을 가져오는 함수

#len(dataset)을 했을 때 데이터셋의 크기를 리턴할 len
#dataset[i]을 했을 때 i번째 샘플을 가져오도록 하는 인덱싱을 위한 get_item
"""

#2. 커스텀 데이터셋(Custom Dataset)으로 선형 회귀 구현하기
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Dataset 상속
class CustomDataset(Dataset):
  def __init__(self):
    self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
    self.y_data = [[152], [185], [180], [196], [142]]

  # 총 데이터의 개수를 리턴
  def __len__(self):
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = torch.nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

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
"""
Epoch    0/20 Batch 1/3 Cost: 29410.156250
Epoch    0/20 Batch 2/3 Cost: 7150.685059
Epoch    0/20 Batch 3/3 Cost: 3482.803467
... 중략 ...
Epoch   20/20 Batch 1/3 Cost: 0.350531
Epoch   20/20 Batch 2/3 Cost: 0.653316
Epoch   20/20 Batch 3/3 Cost: 0.010318
"""

# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]])
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
"""
훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[151.2319]], grad_fn=<AddmmBackward>)
"""