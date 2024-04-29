import torch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)#tensor([0., 1., 2., 3., 4., 5., 6.])

print(t.dim())  # rank. 즉, 차원
#shape나 size()를 사용하면 크기를 확인 가능
print(t.shape)  # shape
print(t.size()) # size
"""
1
torch.Size([7])
torch.Size([7])
"""

print(t[0], t[1], t[-1])  # 인덱스로 접근
print(t[2:5], t[4:-1])    # 슬라이싱
print(t[:2], t[3:])       # 슬라이싱
"""
tensor(0.) tensor(1.) tensor(6.)
tensor([2., 3., 4.]) tensor([4., 5.])
tensor([0., 1.]) tensor([3., 4., 5., 6.])
"""

#2) 2D with PyTorch
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
print(t)
print(t.dim())  # rank. 즉, 차원 #2
print(t.size()) # shape #torch.Size([4, 3])

print(t[:, 1]) # 첫번째 차원을 전체 선택한 상황에서 두번째 차원의 첫번째 것만 가져온다.
print(t[:, 1].size()) # ↑ 위의 경우의 크기
"""
tensor([ 2.,  5.,  8., 11.])
torch.Size([4])
"""

print(t[:, :-1]) # 첫번째 차원을 전체 선택한 상황에서 두번째 차원에서는 맨 마지막에서 첫번째를 제외하고 다 가져온다.
"""
tensor([[ 1.,  2.],
        [ 4.,  5.],
        [ 7.,  8.],
        [10., 11.]])
"""

#3)브로드캐스팅(Broadcasting)
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2) #tensor([[5., 5.]])

# Vector + scalar
"""
 크기가 다른 텐서들 간의 연산을 보겠습니다. 
 아래는 벡터와 스칼라가 덧셈 연산을 수행하는 것을 보여줍니다. 
 물론, 수학적으로는 원래 연산이 안 되는게 맞지만 파이토치에서는 브로드캐스팅을 통해 이를 연산합니다.
"""
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # [3] -> [3, 3]
print(m1 + m2) #tensor([[4., 5.]])

# 2 x 1 Vector + 1 x 2 Vector : 벡터 간 연산에서 브로드캐스팅이 적용되는 경우
#원래 m1의 크기는 (1, 2)이며 m2의 크기는 (1,)
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]]) #m2의 크기는 (1,). 그런데 파이토치는 m2의 크기를 (1, 2)로 변경하여 연산을 수행
"""
# 브로드캐스팅 과정에서 실제로 두 텐서가 아래와 같이 변경됨.
[1, 2]
==> [[1, 2],
     [1, 2]]
[3]
[4]
==> [[3, 3],
     [4, 4]]
"""
print(m1 + m2)
"""
tensor([4., 5.],
       [5., 6.]])
"""

#4) 자주 사용되는 기능들
#4-1) 행렬 곱셈과 곱셈의 차이(Matrix Multiplication Vs. Multiplication) :  행렬 곱셈(.matmul)과 원소 별 곱셈(.mul)
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
"""
[1, 2]   [1]
       *
[3, 4]   [2]
행렬곱은 [1*1+2*2]
        [3*1+4*2]
"""
print(m1.matmul(m2)) # 2 x 1
"""
Shape of Matrix 1:  torch.Size([2, 2])
Shape of Matrix 2:  torch.Size([2, 1])
tensor([[ 5.],
        [11.]])

"""

#element-wise 곱셈 : 동일한 크기의 행렬이 동일한 위치에 있는 원소끼리 곱하는 것
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
"""
[1, 2]   [1]
       *
[3, 4]   [2]
여기서
# 브로드캐스팅 과정에서 m2 텐서가 아래와 같이 변경
[1]
[2]
==> [[1, 1],
     [2, 2]]
이 되므로,
[1*1,2*1]
[3*2,4*2]
"""
print(m1 * m2) # 2 x 2
print(m1.mul(m2))
"""
Shape of Matrix 1:  torch.Size([2, 2])
Shape of Matrix 2:  torch.Size([2, 1])
tensor([[1., 2.],
        [6., 8.]])
tensor([[1., 2.],
        [6., 8.]])
"""

#4-2) 평균(Mean)
t = torch.FloatTensor([1, 2])
print(t.mean()) #tensor(1.5000)

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.mean())#tensor(2.5000) #4개의 원소의 평균인 2.5

#dim=0이라는 것은 첫번째 차원을 의미. 행렬에서 첫번째 차원은 '행'을 의미
print(t.mean(dim=0)) #tensor([2., 3.])
"""
인자로 dim을 준다면 해당 차원을 제거한다는 의미.
다시 말해 행렬에서 '열'만을 남기겠다는 의미.
기존 행렬의 크기는 (2, 2)였지만 이를 수행하면 열의 차원만 보존되면서 (1, 2)가 됨. 이는 (2,)와 같으며 벡터.
열의 차원을 보존하면서 평균을 구하면 아래와 같이 연산합니다.
"""
"""
# 실제 연산 과정
t.mean(dim=0)은 입력에서 첫번째 차원을 제거한다.

[[1., 2.],
 [3., 4.]]

1과 3의 평균을 구하고, 2와 4의 평균을 구한다.
결과 ==> [2., 3.]
"""

#인자로 dim=1을. 이번에는 두번째 차원을 제거
print(t.mean(dim=1))  #tensor([1.5000, 3.5000])
"""
[[1., 2.],
 [3., 4.]]

1과 3의 평균을 구하고, 2와 4의 평균을 구한다.
결과 ==> [2., 3.]

"""
#인자로 dim=-1을. 이번에는 마지막 차원을 제거
#결국 열의 차원을 제거한다는 의미와 같다. 그러므로 위와 출력 결과가 같다.
print(t.mean(dim=-1))  #tensor([1.5000, 3.5000])

#4-3) 덧셈(Sum)
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
"""
tensor([[1., 2.],
        [3., 4.]])
"""
print(t.sum()) # 단순히 원소 전체의 덧셈을 수행 #tensor(10.)
print(t.sum(dim=0)) # 행을 제거 #tensor([4., 6.])
print(t.sum(dim=1)) # 열을 제거 #tensor([3., 7.])
print(t.sum(dim=-1)) # 열을 제거 #tensor([3., 7.])

#4-4) 최대(Max)와 아그맥스(ArgMax)
#최대(Max)는 원소의 최대값을 리턴하고, 아그맥스(ArgMax)는 최대값을 가진 인덱스를 리턴합니다.
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
"""
tensor([[1., 2.],
        [3., 4.]])
"""
print(t.max()) # Returns one value: max #tensor(4.)

print(t.max(dim=0)) # Returns two values: max and argmax #(tensor([3., 4.]), tensor([1, 1]))
"""
행의 차원을 제거한다는 의미이므로 (1, 2) 텐서를 만듭니다. 결과는 [3, 4]입니다.

그런데 [1, 1]이라는 값도 함께 리턴되었습니다. 
max에 dim 인자를 주면 argmax도 함께 리턴하는 특징 때문입니다. 
첫번째 열에서 3의 인덱스는 1이었습니다. 두번째 열에서 4의 인덱스는 1이었습니다.
그러므로 [1, 1]이 리턴됩니다. 어떤 의미인지는 아래 설명해봤습니다.
"""
"""
# [1, 1]가 무슨 의미인지 봅시다. 기존 행렬을 다시 상기해봅시다.
[[1, 2],
 [3, 4]]
첫번째 열에서 0번 인덱스는 1, 1번 인덱스는 3입니다.
두번째 열에서 0번 인덱스는 2, 1번 인덱스는 4입니다.
다시 말해 3과 4의 인덱스는 [1, 1]입니다.
"""

print('Max: ', t.max(dim=0)[0]) #Max:  tensor([3., 4.])
print('Argmax: ', t.max(dim=0)[1]) #Argmax:  tensor([1, 1])

print(t.max(dim=1)) #(tensor([2., 4.]), tensor([1, 1]))
print(t.max(dim=-1)) #tensor([1, 1]))


