'''
@author: Dzh
@date: 2019/12/9 16:35
@file: partial_training.py
'''
import torch
import torch.utils.data as Data
import math

# 每一批训练的数据个数
BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# 把x和y转为TensorDataset类型
torch_dataset = Data.TensorDataset(x, y)

'''
创建DataLoader用于分批训练
dataset 是要用来训练的数据，TensorDataset类型
batch_size 表示每一批训练的数据个数
shuffle 表示是否要打乱数据进行训练（打乱比较好）
num_workers 表示使用多少个线程来进行训练，windows上不能添加这个参数
'''
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=2
)

# 打印出数据总共被分为多少批
batch_number = math.ceil(x.size()[0]/BATCH_SIZE)
print('数据总共分为{}批'.format(batch_number))

# 所有的数据，需要进行多少次迭代训练，这里是3表示3次
for epoch in range(3):
    # 分批训练，loader每次释放一批数据出来进行训练
    for step, (batch_x, batch_y) in enumerate(loader):
        '''
        epoch 表示第多少次迭代
        step 表示第多少批数据
        batch_x 表示当前批数对应的x
        batch_y 表示当前批数对应的y
        '''
        print('Epoch:', epoch, '| Step:', step, '| batch_x:', batch_x.numpy(), '| batch_y:', batch_y.numpy())