# -*- coding: utf-8 -*-
"""
Created on Wed May  2 23:40:46 2018
        存储和恢复模型并查看参数
        link : http://www.pytorchtutorial.com/pytorch-note5-save-and-restore-models/
@author: bejin
"""

'''
在模型完成训练后，我们需要将训练好的模型保存为一个文件供测试使用，或者因为一些原因我们需要继续之前的状态训练之前保存的模型，那么如何在PyTorch中保存和恢复模型呢？

参考PyTorch官方的这份repo，我们知道有两种方法可以实现我们想要的效果。
'''

# 方法一(推荐)：
# 第一种方法也是官方推荐的方法，只保存和恢复模型中的参数。
# 保存

import torch
torch.save(the_model.state_dict(), PATH)

# 恢复
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))

# 使用这种方法，我们需要自己导入模型的结构信息。

# 方法二：
# 使用这种方法，将会保存模型的参数和结构信息。
# 保存

torch.save(the_model, PATH)

# 恢复
the_model = torch.load(PATH)

# 一个相对完整的例子
torch.save({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        }, 'checkpoint.tar' )

# loading
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.evaluate, checkpoint['epoch']))


# 获取模型中某些层的参数
# 对于恢复的模型，如果我们想查看某些层的参数，可以：

# 定义一个网络
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 20, 5)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20, 64, 5)),
            ('relu2', nn.ReLU())
        ]))

# 打印网络的结构
print(model)


# 如果我们想获取conv1的weight和bias：
params = model.state_dict()
for k, v in params.items():
    print(k)    # 打印网络中的变量名
print(params['conv1.weight'])
print(params['conv1.bias'])


