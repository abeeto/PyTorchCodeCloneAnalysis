# coding:utf8
import warnings
import torch as t


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port = 8097  # visdom 端口
    model = 'SqueezeNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = 'E:/SOURCE/kaggle-GogCat/train'  # 训练集存放路径
    test_data_root = 'E:/SOURCE/kaggle-GogCat/test1'   # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.005  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

'''
class A(object):
    def __init__(self):
        print("万俊峰好帅")

def function():
    return 0
'''

if __name__ == '__main__':
    new_config = {'lr':0.1}
    opt._parse(new_config)

    # a = A()
    # a.device = t.device('cuda')
    #
    # function.device = t.device('cuda')

    # b = 3.1415
    # b.device = t.device('cpu')