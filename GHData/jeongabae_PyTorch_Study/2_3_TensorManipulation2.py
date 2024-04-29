import numpy as np
import torch
"""
view(), squeeze(), unsqueeze()는 텐서의 원소 수를 그대로 유지하면서 모양과 차원을 조절합니다.
"""
#4) 뷰(View) - 원소의 수를 유지하면서 텐서의 크기 변경. 매우 중요함!!
#파이토치 텐서의 뷰(View)는 넘파이에서의 리쉐이프(Reshape)와 같은 역할을 합니다.
"""
view는 기본적으로 변경 전과 변경 후의 텐서 안의 원소의 개수가 유지되어야 합니다.
파이토치의 view는 사이즈가 -1로 설정되면 다른 차원으로부터 해당 값을 유추합니다.
"""
# Reshape라는 이름에서 알 수 있듯이, 텐서의 크기(Shape)를 변경해주는 역할
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape) #torch.Size([2, 2, 3])

#4-1) 3차원 텐서에서 2차원 텐서로 변경
print(ft.view([-1, 3])) # ft라는 텐서를 (?, 3)의 크기로 변경 #-1은 첫번째 차원은 사용자가 잘 모르겠으니 파이토치에 맡기겠다는 의미
                        #3은 두번째 차원의 길이는 3을 가지도록 하라는 의미
                        #내부적으로 크기 변환은 다음과 같이 이루어짐. (2, 2, 3) -> (2 × 2, 3) -> (4, 3)
"""
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]])
"""
print(ft.view([-1, 3]).shape) #torch.Size([4, 3])

""" 규칙 정리
view는 기본적으로 변경 전과 변경 후의 텐서 안의 원소의 개수가 유지되어야 합니다.
파이토치의 view는 사이즈가 -1로 설정되면 다른 차원으로부터 해당 값을 유추합니다.
"""

#4-2) 3차원 텐서의 크기 변경

#아래의 예에서 (2 × 2 × 3) = (? × 1 × 3) = 12를 만족해야 하므로 ?는 4가 됩니다.
print(ft.view([-1, 1, 3]))
"""
tensor([[[ 0.,  1.,  2.]],

        [[ 3.,  4.,  5.]],

        [[ 6.,  7.,  8.]],

        [[ 9., 10., 11.]]])
"""
print(ft.view([-1, 1, 3]).shape) #torch.Size([4, 1, 3])

#5) 스퀴즈(Squeeze) - 1인 차원을 제거한다.
"""
스퀴즈는 차원이 1인 경우에는 해당 차원을 제거합니다.
실습을 위해 임의로 (3 × 1)의 크기를 가지는 2차원 텐서를 만들겠습니다.
"""
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
"""
tensor([[0.],
        [1.],
        [2.]])
"""
print(ft.shape)#torch.Size([3, 1])

#스퀴즈(Squeeze) - 1인 차원을 제거한다.
print(ft.squeeze()) #tensor([0., 1., 2.]) #두번째 차원이 1이므로 squeeze를 사용하면 (3,)의 크기를 가지는 텐서로 변경
print(ft.squeeze().shape) #torch.Size([3]) # 1이었던 두번째 차원이 제거되면서 (3,)의 크기를 가지는 텐서로 변경되어 1차원 벡터가 된 것

#6) 언스퀴즈(Unsqueeze) - 특정 위치에 1인 차원을 추가한다.
ft = torch.Tensor([0, 1, 2])
print(ft.shape) #torch.Size([3])

#현재는 차원이 1개인 1차원 벡터입니다. 여기에 첫번째 차원에 1인 차원을 추가해보겠습니다.
# 첫번째 차원의 인덱스를 의미하는 숫자 0을 인자로 넣으면 첫번째 차원에 1인 차원이 추가됩니다.
print(ft.unsqueeze(0)) #tensor([[0., 1., 2.]]) # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
print(ft.unsqueeze(0).shape) #torch.Size([1, 3])
"""뷰로도 위와 같은 연산 가능
print(ft.view(1, -1)) #tensor([[0., 1., 2.]])
print(ft.view(1, -1).shape) #torch.Size([1, 3])
"""

print(ft.unsqueeze(1)) #unsqueeze의 인자로 1을 넣어보겠습니다. 인덱스는 0부터 시작하므로 이는 두번째 차원에 1을 추가하겠다는 것을 의미
print(ft.unsqueeze(1).shape)
"""
tensor([[0.],
        [1.],
        [2.]])
torch.Size([3, 1]) #현재 크기는 (3,)이었으므로 두번째 차원에 1인 차원을 추가하면 (3, 1)의 크기를 가지게 됩니다. 
"""

print(ft.unsqueeze(-1)) #-1은 인덱스 상으로 마지막 차원을 의미합니다. 현재 크기는 (3,)
print(ft.unsqueeze(-1).shape)
"""
tensor([[0.],
        [1.],
        [2.]])
torch.Size([3, 1]) #마지막 차원에 1인 차원을 추가하면 (3, 1)의 크기를 가지게 됩니다. 
"""

#7) 타입 캐스팅(Type Casting) : 자료형을 변환하는 것
lt = torch.LongTensor([1, 2, 3, 4]) #long 타입의 lt라는 텐서를 선언
print(lt) #tensor([1, 2, 3, 4])
print(lt.float())#텐서에다가 .float()를 붙이면 바로 float형으로 타입이 변경됨 #tensor([1., 2., 3., 4.])

bt = torch.ByteTensor([True, False, False, True]) # Byte 타입의 bt라는 텐서
print(bt) #tensor([1, 0, 0, 1], dtype=torch.uint8)
print(bt.long()) #tensor([1, 0, 0, 1])
print(bt.float()) #tensor([1., 0., 0., 1.])

#8) 두 텐서 연결하기(concatenate)

# 1. (2 × 2) 크기의 텐서를 두 개 만듭니다.
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

#2. 두 텐서를 torch.cat([ ])를 통해 연결
print(torch.cat([x, y], dim=0)) #dim=0은 첫번째 차원을 늘리라는 의미
"""dim=0을 인자로 했더니 두 개의 (2 × 2) 텐서가 (4 × 2) 텐서가 된 것을 볼 수 있습니다.
tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])
"""

print(torch.cat([x, y], dim=1))
"""dim=1을 인자로 했더니 두 개의 (2 × 2) 텐서가 (2 × 4) 텐서가 된 것을 볼 수 있습니다.
tensor([[1., 2., 5., 6.],
        [3., 4., 7., 8.]])
"""

#9) 스택킹(Stacking) - 연결(concatenate)을 하는 또 다른 방법

#크기가 (2,)로 모두 동일한 3개의 벡터
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

# torch.stack을 통해서 3개의 벡터를 모두 스택킹
print(torch.stack([x, y, z])) #print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))와 같음
"""3개의 벡터가 순차적으로 쌓여 (3 × 2) 텐서가 된 것
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
"""

#스택킹에 추가적으로 dim을 인자로 줄 수도 있습니다. 이번에는 dim=1 인자를 주겠습니다.
#이는 두번째 차원이 증가하도록 쌓으라는 의미로 해석할 수 있습니다.
print(torch.stack([x, y, z], dim=1))
"""
tensor([[1., 2., 3.],
        [4., 5., 6.]])
"""

#10) ones_like와 zeros_like - 0으로 채워진 텐서와 1로 채워진 텐서
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
"""
tensor([[0., 1., 2.],
        [2., 1., 0.]])
"""
print(torch.ones_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 1로 채우기
"""
tensor([[1., 1., 1.],
        [1., 1., 1.]])
"""
print(torch.zeros_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 0으로 채우기
"""
tensor([[0., 0., 0.],
        [0., 0., 0.]])
"""

#11) In-place Operation (덮어쓰기 연산) -  연산 뒤에 _를 붙이면 기존의 값을 덮어쓰기 합니다.
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.)) # 곱하기 2를 수행한 결과를 출력
"""
tensor([[2., 4.],
        [6., 8.]])
"""
print(x) # 기존의 값 출력
"""
tensor([[1., 2.],
        [3., 4.]])
"""

#그런데 연산 뒤에 _를 붙이면 기존의 값을 덮어쓰기 합니다.
print(x.mul_(2.))  # 곱하기 2를 수행한 결과를 변수 x에 값을 저장하면서 결과를 출력
print(x) # 기존의 값 출력
"""
tensor([[2., 4.],
        [6., 8.]])
tensor([[2., 4.],
        [6., 8.]])
"""
