import numpy as np
import matplotlib.pyplot as plt

# 构建训练数据集 x,y分别对应输入和输入，相同的索引对应相同的输入和输出
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
# 定义目标模型  y = x * w
def forward(x):
    return x*w
 
# 定义误差函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
 
# 穷举法
# 定义权重的list，存放权重
w_list = []
# 定义MSE的list，用于存放权重对应的误差值
mse_list = []
for w in np.arange(0.0, 4.1, 0.1): #设置间隔为0.1，取0.0,0.1……到4.0之间的数组作为w
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)
    
# 使用plt绘制图形
plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()    