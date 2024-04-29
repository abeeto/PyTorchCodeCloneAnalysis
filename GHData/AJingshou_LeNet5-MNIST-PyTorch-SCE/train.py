import matplotlib.pyplot as plt
import numpy as np
import tqdm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import argparse
import functools
import os
import time

from model import Model
import numpy as np
import torch
from torchvision.datasets import mnist
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from loss import SCELoss
from loss import Bootstrapping
import labelnoise

# 定义超参数
'''
asym = True  # 是否使用非对称噪声
noise_fractions = 0.35   # 噪声比例
batch_size = 64  # 一次训练的样本数目
learning_rate = 0.0001  # 学习率
iteration_num = 50  # 迭代次数
network = LeNet5()  # 实例化网络
optimizer = optim.SGD(params=network.parameters(), lr=learning_rate, momentum=0.78)
alpha = 0.1      # SCE方法的参数alpha
bootstarp_beta = 0.95    # Bootstarp方法的参数Beta
loss_model = 'SCE'      #确定损失函数类型
criterion = nn.CrossEntropyLoss()   # 初始化定义损失函数
'''
#[i for i in range] 列表解析式
'''
def get_noise_mapping(seed=1):
  np.random.seed(seed)
  noise_mapping = np.array([i for i in range(10)])
  np.random.shuffle(noise_mapping)
  return noise_mapping

def noisify_labels(y, noise_fractions, noise_mapping, seed=1):
  
  np.random.seed(seed)
  y_noisy = np.zeros(y.shape)
  noisy_sum = 0
  clean_sum = 0
  for i in range(y.shape[0]):
    if np.random.rand() <= noise_fractions[y[i]]:
      #print('%d -> %d' % (y_train[i], noise_mapping[y_train[i]]))
      y_noisy[i] += noise_mapping[y[i]]
      noisy_sum += 1
    else:
      #print('%d -> %d' % (y_train[i], y_train[i]))
      y_noisy[i] += y[i]
      clean_sum += 1
  #print(noisy_sum, clean_sum, noisy_sum/float(y.shape[0]), clean_sum/float(y.shape[0]))
  return y_noisy

def noisify_mnist(noise_fraction):
  # load the data
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = np.reshape(x_train, (x_train.shape[0], 784)) / 255.
  x_test = np.reshape(x_test, (x_test.shape[0], 784)) / 255.
  
  noise_mapping = get_noise_mapping()
  noise_fractions = [noise_fraction for i in range(10)]
  
  print()
  print(noise_mapping)
  print(np.round(noise_fractions, 3))
  
  map = np.zeros((10,10))
  for i in range(10):
    map[i,noise_mapping[i].astype('int32')] += 1
  
  y_train_noisy = noisify_labels(y_train, noise_fractions, noise_mapping)
  return x_train, y_train, y_train_noisy, x_test, y_test, map, noise_mapping
'''
# 正则化
def normalize(x):
    x = 2 * ((x * 255. / 256.) - .5)
    x += torch.zeros_like(x).uniform_(0, 1. / 128)
    return x

def plotdistribution(Label,Mat):
    """
    :param Label: 点的类别标签
    :param Mat: 二维点坐标矩阵
    :return:
    """
    tsne = TSNE(n_components=2, random_state=0)
    Mat = tsne.fit_transform(Mat[:])

    x = Mat[:, 0]
    y = Mat[:, 1]
    # map_size = {0: 5, 1: 5}
    # size = list(map(lambda x: map_size[x], Label))
    map_color = {0: 'r', 1: 'g',2:'b',3:'y',4:'k',5:'m',6:'c',7:'pink',8:'grey',9:'blueviolet'}
    color = list(map(lambda x: map_color[x], Label))
    # 代码会出错，因为marker参数不支持列表
    # map_marker = {-1: 'o', 1: 'v'}
    # markers = list(map(lambda x: map_marker[x], Label))
    #  plt.scatter(np.array(x), np.array(y), s=size, c=color, marker=markers)
    # 下面一行代码为修正过的代码
    px.scatter(np.array(x), np.array(y), s=5, c=color, marker='o')  # scatter函数只支持array类型数据


"""显示数据"""
def plot_embedding(result, label, title):   #传入1083个2维数据，1083个标签，图表标题
    x_min, x_max = np.min(result, 0), np.max(result, 0)   #分别求出每一列最小值和最大值
    data = (result - x_min) / (x_max - x_min)   #将数据进行正则化，分母为数据的总长度，因此分子一定小于分母，生成的矩阵元素都是0-1区间内的
    plt.figure()   #创建一个画布
    for i in range(data.shape[0]):   #遍历所有的数据，共1083个
        plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(label[i] / 10.))
    plt.title(title)   #设置标题
    plt.show()
    
if __name__ == '__main__':
    batch_size = 256
    Dataset = functools.partial(labelnoise.MNISTNoisyLabels,
                                   noise_type='asymmetric',
                                   noise_rate=0.4,
                                   seed=12345)
    train_dataset = Dataset(root='./train',
                      train=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Lambda(normalize)
                      ]))
    '''
    test_dataset = Dataset(root='./test',
                  train=False,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Lambda(normalize)
                  ]))
    '''
    #                  download=True,
    #train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Lambda(normalize)]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = Model()
    sgd = SGD(model.parameters(), lr=1e-1,weight_decay=1e-4)
    loss_fn = CrossEntropyLoss()
    #loss_fn = SCELoss(alpha=0.1, beta=1.0, num_classes=10)
    #loss_fn = Bootstrapping(num_classes=10, t=0.05)
    all_epoch = 100
    acc_array = np.array([])
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            sgd.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        #model.load_state_dict(torch.load("./my_model_epoch1500.pth"))
        device = torch.device("cpu")
        y_np = []
        z_np = []
        predict_index = [0,0,0,0,0,0,0,0,0,0]
        correct_num_index = [0,0,0,0,0,0,0,0,0,0]
        #all_correct_num_index = [0,0,0,0,0,0,0,0,0,0]
        #all_sample_num_index = [0,0,0,0,0,0,0,0,0,0]
        predict_label = np.array([])
        
        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float()).detach()
            predict_y = np.argmax(predict_y, axis=-1)
            predict_label = np.append(predict_label, predict_y)
            current_correct_num = predict_y == test_label
            #dim0 = predict_y.shape
            for ii in range(16):
                correct_num_index[int(test_label[ii])] = correct_num_index[int(test_label[ii])] + 1
                if predict_y[ii] == test_label[ii]:
                    predict_index[int(test_label[ii])] = predict_index[int(test_label[ii])] + 1
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
            #test_x = test_x.to(device).view(-1, 784)
            '''
            x_reconst, mu, log_var, z = model(test_x)
            y_cpu = test_label.cpu().detach().numpy()
            z_cpu = z.cpu().detach().numpy()
            y_np.extend(y_cpu)
            z_np.extend(z_cpu)
            reconst_loss = F.binary_cross_entropy(x_reconst, test_x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            if (idx + 1) % 10 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}".format(0 + 1, 1, idx + 1, len(test_loader), reconst_loss.item(), kl_div.item()))
            '''    
        #encoded_samples = []
        #print(predict_label)
        predict_label_tensor = torch.from_numpy(predict_label)
        #eval_label = y_np
        #eval_data = z_np
        #plotdistribution(eval_label,eval_data)
        
        
        #fig = px.scatter(result, x=0, y=1, color=predict_label.astype(str),labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
        
        if current_epoch == 99:
            tsne = TSNE(n_components=2, init='pca', random_state=0)   #n_components将64维降到该维度，默认2；init设置embedding初始化方式，可选pca和random，pca要稳定些
            #t0 = time()   #记录开始时间
            tsne_data = torch.flatten(test_dataset.data, start_dim=1, end_dim=2)
            result = tsne.fit_transform(tsne_data)   #进行降维，[1083,784]-->[1083,2]
            fig = plot_embedding(result, predict_label,'t-SNE embedding of the digits')
            plt.show(fig)
            
        acc = all_correct_num / all_sample_num
        acc_array = np.append(acc_array, acc)
        for iii in range(10):
            print('accuracy{}: {:.4f}'.format(iii , predict_index[iii] / correct_num_index[iii]))
        # acc_array是每个epoch训练结果组合的序列
        print('accuracy: {:.2f}'.format(acc))
#        torch.save(model, 'models/mnist_{:.2f}.pkl'.format(acc))
