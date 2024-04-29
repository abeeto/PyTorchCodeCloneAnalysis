import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # GPU 환경용

# batch_size은 배치 크기, input_Dim은 입력의 차원
# hidden_Dim는 은닉층의 차원이며, output_Dim은 출력의 차원

# Batch X
batch_size, input_Dim, hidden_Dim, output_Dim = 1, 4, 2, 1

# Batch로 돌리는 경우
#batch_size, input_Dim, hidden_Dim, output_Dim = 10, 4, 2, 1 

# 난수 생성시  난수의 최댓값, 최솟값 설정
high, low = 5, 1

# 무작위성 배제를 위한 seed 설정
torch.manual_seed(777)

# 무작위로 입력과 출력 데이터 생성.
# input = 4 X 4 , output = 4 X 1
input = torch.randint(low, high, (batch_size, input_Dim), device=device, dtype=dtype)
output = torch.randint(low, high, (batch_size, output_Dim), device=device, dtype=dtype)

# 무작위로 가중치를 초기화.
w1 = torch.randint(low, high, (input_Dim, hidden_Dim), device=device, dtype=dtype)
w2 = torch.randint(low, high, (hidden_Dim, output_Dim), device=device, dtype=dtype)

print('input :\n', input) # 4 X 4 
print('output :\n', output) # 4 X 1
print('w1 :\n', w1) # 4 X 2
print('w2 :\n', w2) # 2 X 1

'''
원소는 float 형태

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
    h = input.mm(w1) # mm == mat multiply
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    
    print('h :\n', h)
    print('h_relu :\n', h_relu)
    print('y :\n', output)
    print('y_pred :\n', y_pred)
    
    # (예측값 - 실제값)**2 --> 모든 원소 평균 구한 뒤(tensor형태), 값(value) 형태로 return
    # .mean(dim=None, keepdim=False)
    loss = (y_pred - output).pow(2).mean().item()

    if t % 100 == 99:
        print(t, loss)
    else:
        print('횟수 :',t)
        print('Loss :', loss)

    # loss에 따른 w1, w2의 미분값을 계산하고 weight 업데이트
    grad_y_pred = 2.0 * (y_pred - output) # 예측값에 대한 loss의 미분값
    print('grad_y_pred :\n', grad_y_pred)
    grad_w2 = h_relu.t().mm(grad_y_pred) # w2로의 역전파
    print('grad_w2 :\n', grad_w2)
    grad_h_relu = grad_y_pred.mm(w2.t()) # h_relu로의 역전파
    print('grad_h_relu :\n', grad_h_relu)
    grad_h = grad_h_relu.clone() 
    print('grad_h :\n', grad_h)
    grad_h[h < 0] = 0 # 활성화 함수가 relu니까 0보다 작으면 0
    print('grad_h[h < 0] :\n', grad_h[h < 0])
    grad_w1 = input.t().mm(grad_h) # w1로의 역전파
    print('grad_w1 :\n', grad_w1)


    # loss가 작아지는 방향으로 weight update
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    print('updated w1 :\n', w1)
    print('updated w2 :\n', w2)
