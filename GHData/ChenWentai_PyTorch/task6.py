#构造数据：J为损失函数，J_prime为损失函数的导数
import numpy as np
import matplotlib.pyplot as plot
J = lambda w1, w2: w1**2+10*w2**2
J_prime1 = lambda w1: 2*w1
J_prime2 = lambda w2: 2*w2
w1 = 1
w2 = -1
epoch = 200
lr = 0.1

#SGD
Loss_sgd = []
W1_sgd = []
W2_sgd = []
for i in range(epoch):

    w1 -= lr*J_prime1(w1)
    
    w2 -= lr*J_prime2(w2)
    
    W1_sgd.append(w1)
    W2_sgd.append(w2)
    Loss_sgd.append(J(w1, w2))

# Momentum
gamma = 0.5
v1 = 0
v2 = 0
s = 0
Loss_moment = []
W1_moment = []
W2_moment = []
for i in range(epoch):
    v1 = gamma*v1 + lr*J_prime1(w1)
    w1 -= v1
    v2 = gamma*v2 + lr*J_prime2(w2)
    w2 -= v2    
    W1_moment.append(w1)
    W2_moment.append(w2)
    Loss_moment.append(J(w1, w2))

#Adagrad
v = 0
s = 0
Loss_ada = []
W1_ada = []
W2_ada = []
s1=s2=0
for i in range(epoch):
    s1 += J_prime1(w1)**2
    w1 -= lr*(J_prime1(w1)/np.sqrt(s1))
    s2 += J_prime2(w2)**2
    w2 -= lr*(J_prime2(w2)/np.sqrt(s2))
    
    W1_ada.append(w1)
    W2_ada.append(w2)
    Loss_ada.append(J(w1, w2))

    #RMSprop
epoch = 200
lambda0 = 0.01
gamma = 0.5
v = 0
s = 0
Loss_RMS = []
W1_RMS = []
W2_RMS = []
s1=s2=0
for i in range(epoch):
    s1 = gamma*s1 + (1-gamma)*(J_prime1(w1)**2)
    
    w1 -= lambda0*(J_prime1(w1)/np.sqrt(s1))
    s2 = gamma*s2 + (1-gamma)*(J_prime2(w2)**2)
    w2 -= lambda0*(J_prime2(w2)/np.sqrt(s2))
    
    
    W1_RMS.append(w1)
    W2_RMS.append(w2)
    Loss_RMS.append(J(w1, w2))
#画出loss和weight的曲线
LOSS = [Loss_sgd, Loss_moment, Loss_ada, Loss_RMS]
labels = ['SGD', 'Momentum','Adagrad','RMSprop']
for i, loss in enumerate(LOSS):
    plt.plot(loss, label=labels[i])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.savefig('./task6/Loss.jpg', dpi=500)
plt.show()

W1 = [W1_sgd, W1_moment, W1_ada, W1_RMS]
for i, w1 in enumerate(W1):
    plt.plot(w1, label=labels[i])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('W1')
plt.title('W1')
plt.savefig('./task6/W1.jpg', dpi=500)
plt.show()

W2 = [W2_sgd, W2_moment, W2_ada, W2_RMS]
for i, w2 in enumerate(W2):
    plt.plot(w2, label=labels[i])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('W2')
plt.title('W2')
plt.savefig('./task6/W2.jpg', dpi=500)
plt.show()