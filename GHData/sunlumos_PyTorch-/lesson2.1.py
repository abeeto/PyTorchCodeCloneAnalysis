import matplotlib.pyplot as plt

# todo 随机取值梯度下降 

# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
# initial guess of weight 设置初试的权重猜测
w = 1.0
 
# define the model linear model y = w*x
def forward(x):
    return x*w
 
#define the cost function MSE 计算
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2
 
# define the gradient function  求梯度 随机取值求梯度
def gradient(x,y):
    return 2*x*(x*w - y) 
 
print('predict (before training)', 4, forward(4))
# 进行100轮训练
for epoch in range(100):
    for x , y in zip(x_data, y_data):
       grad = gradient(x, y)
       w = w - 0.01 * grad
       print("\tgrad:", x, y, grad)
       l = loss(x, y)
    
    print("epoch:", epoch, "w = ", w, "loss=", l)

print('Predict(after training)', 4, forward(4))

# 普通梯度下降性能低但时间低  随机梯度下降性能高但时间花费高
# 因此在正式算法中一般使用batch 小批量的随机梯度下降