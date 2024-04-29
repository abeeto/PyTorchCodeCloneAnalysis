import os
import sys
import torch
# 把我们写的模型导入进来
from models import vgg, resnet, googlenet
# 把我们的数据导入进来
from data_process import data_process
# 获取当前路径，查看当前路径下是否存在weight目录，为了将weight/下的参数导入进来
path = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))[0]


# /*************** Task 4 : 模型测试***************
# 选择要测试的模型
name = 'vgg'
offset = '_done'  # 这个是我在服务器上训练好的模型参数，vgg/resnet/googlenet均达到90+
if name == 'vgg':
    net = vgg()
if name == 'resnet':
    net = resnet()
if name == 'googlenet':
    net = googlenet()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 将模型拷贝到GPU或CPU上
net.to(device)
trainloader, testloader = data_process()

# 在我们的实验中，测试集和验证集都是一样的，对训练来说我们是需要进行反向传播和参数更新的，但对验证和测试来说，都不需要
# 测试和验证在逻辑上是一模一样的，在代码上有一点点的区别：那就是模型参数
# 每一个训练epoch后面都跟着一个验证，因为我需要判断这一次训练epoch是否accuracy是最高的，如果是最高的我就保存参数（还有其他用途，通过验证的loss和acc判断模型训练到哪一个阶段了）。
# 所以说，每一次验证过程的模型里面的参数都是上一次训练得到的参数
# 但对测试来说，我需要将最优的参数导入进去，然后拿去参加比赛或投入应用
def test():
    # 导入预训练模型的代码千篇一律，网上很多，不做赘述
    pretrain_dict = torch.load(path+'/weight/'+name+offset+'.pkl', map_location=torch.device('cpu'))
    model_dict = {}
    state_dict = net.state_dict()
    for k, v in pretrain_dict.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    # 导入预训练模型参数
    net.load_state_dict(state_dict)

    # 设为验证模式
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # 将数据拷贝到CPU或GPU上
            inputs, targets = inputs.to(device), targets.to(device)
            # 模型预测
            outputs = net(inputs)
            # 求准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('Acc: %.3f%%' % (accuracy))
