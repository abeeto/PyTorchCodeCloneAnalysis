import matplotlib.pyplot as plt
 
# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
# initial guess of weight 设置初试的权重猜测
w = 1.0
 
# define the model linear model y = w*x
def forward(x):
    return x*w
 
#define the cost function MSE 计算
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs,ys):
        y_pred = forward(x)  #求猜测的y值
        cost += (y_pred - y)**2  #求（y^ - y）2  然后加到cost
    return cost / len(xs)  #求平均cost
 
# define the gradient function  求梯度：求和再求均值
def gradient(xs,ys):
    grad = 0
    for x, y in zip(xs,ys):
        grad += 2*x*(x*w - y)   #求均值的公式，2*x*(x*w - y) 
    return grad / len(xs)
 
epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))
# 进行100轮训练
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w-= 0.01 * grad_val  # 0.01 learning rate 是进步率  通过这一步不停地修正w的值来使得预测结果越来越准
    #如果最后曲线发散了，通常原因为学习率取的过大，需要降低
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)
 
print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show() 