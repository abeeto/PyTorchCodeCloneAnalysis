#앞으로의 코드에서는 아래 세 줄의 코드가 이미 진행되었다고 가정
import torch
import torch.nn.functional as F

torch.manual_seed(1)

#1. 파이토치로 소프트맥스의 비용 함수 구현하기 (로우-레벨)

#3개의 원소를 가진 벡터 텐서를 정의
z = torch.FloatTensor([1, 2, 3])

#이 텐서를 소프트맥스 함수의 입력으로 사용하고, 그 결과를 확인
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
"""3개의 원소의 값이 0과 1사이의 값을 가지는 벡터로 변환된 것 확인 가능
tensor([0.0900, 0.2447, 0.6652])
"""
hypothesis.sum()#tensor(1.) #총 원소의 값의 합은 1

# 아래 코드부터는 비용 함수를 직접 구현

# 임의의 3 × 5 행렬의 크기를 가진 텐서
z = torch.rand(3, 5, requires_grad=True)

#텐서에 대해서 소프트맥스 함수를 적용
hypothesis = F.softmax(z, dim=1)# 각 샘플에 대해서 소프트맥스 함수를 적용하여야 -> 두번째 차원에 대해서 소프트맥스 함수를 적용한다는 의미에서 dim=1
print(hypothesis)
"""각 행의 원소들의 합은 1이 되는 텐서로 변환됨.
tensor([[0.2645, 0.1639, 0.1855, 0.2585, 0.1277],
        [0.2430, 0.1624, 0.2322, 0.1930, 0.1694],
        [0.2226, 0.1986, 0.2326, 0.1594, 0.1868]], grad_fn=<SoftmaxBackward>)
"""
"""
소프트맥스 함수의 출력값은 결국 예측값.
즉, 위 텐서는 3개의 샘플에 대해서 5개의 클래스 중 어떤 클래스가 정답인지를 예측한 결과.
"""

# 각 샘플에 대해서 임의의 레이블 만듦.
y = torch.randint(5, (3,)).long()
print(y) #tensor([0, 2, 1])

#각 레이블에 대해서 원-핫 인코딩을 수행
# 모든 원소가 0의 값을 가진 3 × 5 텐서 생성
y_one_hot = torch.zeros_like(hypothesis) #torch.zeros_like(hypothesis)를 통해 모든 원소가 0의 값을 가진 3 × 5 텐서만듦.(이 텐서는 y_one_hot에 저장이 된 상태)
#scatter의 첫번째 인자로 dim=1에 대해서 수행하라고 알려주고, 세번째 인자에 숫자 1을 넣어주므로서 두번째 인자인 y_unsqeeze(1)이 알려주는 위치에 숫자 1을 넣음.
y_one_hot.scatter_(1, y.unsqueeze(1), 1) # y.unsqueeze(1)를 하면 (3,)의 크기를 가졌던 y 텐서는 (3 × 1) 텐서가 됨
                                         #연산 뒤에 _를 붙이면 In-place Operation (덮어쓰기 연산)
"""
print(y.unsqueeze(1))
실행결과
tensor([[0],
        [2],
        [1]])
"""
print(y_one_hot)
"""
tensor([[1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0.]])
"""

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost) #tensor(1.4689, grad_fn=<MeanBackward1>)

#2. 파이토치로 소프트맥스의 비용 함수 구현하기 (하이-레벨)
#2-1. F.softmax() + torch.log() = F.log_softmax()
# Low level
torch.log(F.softmax(z, dim=1))
# High level
F.log_softmax(z, dim=1)# 파이토치에서는 F.log_softmax()라는 도구를 제공
"""둘 다 출력은 다음과 같음.
tensor([[-1.3301, -1.8084, -1.6846, -1.3530, -2.0584],
        [-1.4147, -1.8174, -1.4602, -1.6450, -1.7758],
        [-1.5025, -1.6165, -1.4586, -1.8360, -1.6776]], grad_fn=<LogSoftmaxBackward>)
"""

#2-2. F.log_softmax() + F.nll_loss() = F.cross_entropy()
"""
 로우-레벨로 구현한 비용 함수
 # Low level
# 첫번째 수식
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean() #tensor(1.4689, grad_fn=<MeanBackward1>)

# 두번째 수식
(y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean() #tensor(1.4689, grad_fn=<MeanBackward0>)

"""
# High level
# 세번째 수식
F.nll_loss(F.log_softmax(z, dim=1), y) #F.nll_loss()를 사용할 때는 원-핫 벡터를 넣을 필요없이 바로 실제값을 인자로 사용
                            #tensor(1.4689, grad_fn=<NllLossBackward>)

# 네번째 수식
F.cross_entropy(z, y) #F.cross_entropy()는 F.log_softmax()와 F.nll_loss()를 포함
                    # #tensor(1.4689, grad_fn=<NllLossBackward>)