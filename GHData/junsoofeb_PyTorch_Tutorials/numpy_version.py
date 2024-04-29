import numpy as np

# batch_size은 배치 크기, input_Dim은 입력의 차원
# hidden_Dim는 은닉층의 차원이며, output_Dim은 출력의 차원

# Batch X
batch_size, input_Dim, hidden_Dim, output_Dim = 1, 4, 2, 1

# Batch로 돌리는 경우
#batch_size, input_Dim, hidden_Dim, output_Dim = 10, 4, 2, 1 

# 난수 생성시  난수의 최댓값, 최솟값 설정
high, low = 5, 1

# 무작위성 배제를 위한 seed 설정
np.random.seed(777)

# 무작위로 입력과 출력 데이터 생성.
# input = 4 X 4 , output = 4 X 1
input = np.random.randint(low, high, size = (batch_size, input_Dim))
output = np.random.randint(low, high, size = (batch_size, output_Dim))


# 무작위로 weight 초기화, bias는 생략
# w1 = 4 X 2, w2 = 2 X 1
w1 = np.random.randint(low, high, size = (input_Dim, hidden_Dim))
w2 = np.random.randint(low, high, size = (hidden_Dim, output_Dim))

print('input :\n', input) # 4 X 4 
print('output :\n', output) # 4 X 1
print('w1 :\n', w1) # 4 X 2
print('w2 :\n', w2) # 2 X 1

'''
input : [[4 4 4 3]]             output : [[4]]

w1 :[[2 4]
    [2 4]
    [1 4]
    [2 3]]

w2 :[[3]
    [1]]
    
h : [[26 57]]                   h_relu : [[26 57]]

y : [[4]]                       y_pred : [[135]]
'''


learning_rate = 0.000001
for t in range(1):
    # 순전파, y_predict 구하는 과정
    h = input.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    print('h :\n', h)
    print('h_relu :\n', h_relu)
    print('y :\n', output)
    print('y_pred :\n', y_pred)

    '''
    loss 계산, MSE 사용
    예) 
    y - y_pred)**2 가 
    [   [105625],
        [72900], 
        [64009], 
        [57121]    ] 일 때,
    
    mean(axis = 0) --> 각 col의 평균을 구한 리스트, 세로로 평균 구한 것 --> [ 74913.75] 
    mean(axis = 1) --> 각 row의 평균을 구한 리스트, 가로로 평균 구한 것 --> [ 105625, 72900, 64009, 57121]
    mean(axis = None) --> 모든 원소들의 평균값 --> 74913.75
    '''
    
    print('횟수 :',t)
    # RMSE
    # loss = np.square(y_pred - output).sum()
    # print('Loss :', loss)

    # MSE
    loss = ((y_pred - output)**2).mean(axis = None)
    print('Loss :', loss)

    # loss에 따른 w1, w2의 미분값을 계산하고 weight 업데이트
    grad_y_pred = 2 * (y_pred - output) # 예측값에 대한 loss의 미분값
    print('grad_y_pred :\n', grad_y_pred)
    grad_w2 = h_relu.T.dot(grad_y_pred) # w2로의 역전파
    print('grad_w2 :\n', grad_w2)
    grad_h_relu = grad_y_pred.dot(w2.T) # h_relu로의 역전파
    print('grad_h_relu :\n', grad_h_relu)
    grad_h = grad_h_relu.copy() # h로의 역전파
    print('grad_h :\n', grad_h)
    grad_h[h < 0] = 0 # 활성화 함수가 relu니까 0보다 작으면 0
    print('grad_h[h < 0] :\n', grad_h[h < 0])
    grad_w1 = input.T.dot(grad_h) # w1로의 역전파
    print('grad_w1 :\n', grad_w1)

    # loss가 작아지는 방향으로 weight update
    w1 = w1 - (learning_rate * grad_w1)
    w2 = w2 - (learning_rate * grad_w2)
    
    print('updated w1 :\n', w1)
    print('updated w2 :\n', w2)
    