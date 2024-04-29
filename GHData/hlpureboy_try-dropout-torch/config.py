# 定义超参数
input_size = 28 * 28

num_classes = 2

dataset_path = 'F:/BaiDuNet/medmnist/pneumoniamnist.npz'
datset_name = 'pneumonia'

baselines = [input_size, 1024, 512, 256, 128, 64, num_classes]
# baselines = [input_size, 700, 600, 600,600,512,300,200,100,10]
increase_dropouts = [0, 0.1, 0.3, 0.5, 0.7, 0.9]

reduce_dropouts = increase_dropouts[::-1]

avg_dropouts = [(sum(increase_dropouts) / len(increase_dropouts)) for i in increase_dropouts]

num_epochs = 50
batch_size = 100
learning_rate = 0.00001
