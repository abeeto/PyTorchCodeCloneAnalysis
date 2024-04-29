import torch as t
from matplotlib import pyplot as plt
from IPython import display
t.manual_seed(1000) # 设置随机种子
def get_fake_data(batch_size = 8):
    x = t.randn(batch_size,1)*20 # 随机数。batch_size行 1列
    y = x*2+ (1+t.randn(batch_size,1))*3
    return x,y

x,y = get_fake_data()
plt.scatter(x.squeeze().numpy(),y.squeeze().numpy())

w = t.randn(1,1)
b = t.zeros(1,1)
lr = 0.0001           # 学习率

for ii in range(20000):
    x,y = get_fake_data()

    #forward: 计算loss
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5*(y_pred-y)**2 # 均方误差
    loss = loss.sum()

    # backward: 手动计算
    dloss = 1
    dy_pred = dloss*(y_pred - y)

    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()

    #更新参数
    w.sub_(lr*dw)
    b.sub_(lr*db)

    if(ii%1000 == 0):
        #画图
        display.clear_output(wait=True)  #清除一个单元格的输出
        x = t.arange(0,20).view(-1,1).float() # -1代表不确定的数,注意 t.arange的输出结果为 LongTensor
        y = x.mm(w)+b.expand_as(x)
        print("----------------------------",x,y)
        plt.plot(x.numpy(),y.numpy()) # predicted

        x2,y2 = get_fake_data(batch_size=20)
        plt.scatter(x2.numpy(),y2.numpy()) # true data

        plt.xlim(0,20)
        plt.ylim(0,41)
        plt.show()
        plt.pause(0.5)

print(w.squeeze().item(),b.squeeze().item())

