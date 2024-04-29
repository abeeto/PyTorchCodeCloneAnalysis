'''
    如何在 PyTorch 中设定学习率衰减（learning rate decay）

    link = http://www.pytorchtutorial.com/pytorch-learning-rate-decay/
'''

'''
很多时候我们要对学习率（learning rate）进行衰减，下面的代码示范了如何每30个epoch按10%的速率衰减：
'''

def adjust_learning_rate(optimizer,epoch):
    """ Sets the learning rate to the initial LR decayed by 10 every 30 epochs """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 什么是param_groups?
'''
optimizer通过param_group来管理参数组.param_group中保存了参数组及其对应的学习率,动量等等.所以我们可以通过更改param_group[‘lr’]的值来更改对应参数组的学习率。
'''
import torch.optim as optim
# 有两个`param_group`即,len(optim.param_groups)==2
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
 
#一个参数组
optim.SGD(model.parameters(), lr=1e-2, momentum=.9)

