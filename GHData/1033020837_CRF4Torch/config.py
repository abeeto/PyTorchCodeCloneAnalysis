"""
配置文件
"""

bert_model = './bert-base-chinese/' # bert预训练模型存储路径
batch_size = 32    # 批大小
bert_lr = 1e-4   # bert部分学习率大小
lr = 1e-2   # 非bert部分学习率大小
n_epochs = 20   # 训练轮次
use_lstm = True # 是否在bert后接BILSTM
lstm_hidden_size = 256 # LSTM隐藏层节点数
max_len = 256  # 句子最大长度
output_dir = 'checkpoints/' # 输出模型文件、日志文件的目录
train_file = 'data/example.train'  # 训练文件路径
dev_file = 'data/example.dev'  # 验证文件路径
test_file = 'data/example.test'    # 测试文件路径
use_cuda = False # 是否使用CUDA加速
output_loss_freq = 50 # 每隔多少个step输出loss信息
early_stop = 3 # 早停，若验证集F1多次未提升则提前停止训练
dropout = 0

if use_cuda:
    device = 'cuda'
else:
    device = 'cpu'
