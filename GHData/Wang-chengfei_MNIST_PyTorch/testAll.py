import torch
import LeNet5
from torchvision import datasets,transforms
import os


# 加载模型
net = LeNet5.LeNet5()
if os.path.exists('./MNIST_model.ph'):
    net.load_state_dict(torch.load('./MNIST_model.ph'))
    print('模型加载成功！')
else:
    print('暂时没有模型文件，请先训练模型后再测试！')
    exit()

# 加载测试数据
test_set = datasets.MNIST('./data', download=True, train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=100, num_workers=0)

if not os.path.exists('./error'):
    os.mkdir('./error')

correct = 0
# 测试并显示识别错误的图片
for batch_idx, (data, target) in enumerate(test_loader):
    data = data.float()
    pred = net(data)
    pred = torch.argmax(pred, dim=1)
    correct += torch.eq(pred, target).sum().item()
    image_data = data[torch.eq(pred, target) == False]
    pred_data = pred[torch.eq(pred, target) == False]
    target_data = target[torch.eq(pred, target) == False]
    for i in range(len(image_data)):
        pred_item = pred_data[i].item()
        target_item = target_data[i].item()
        print("预测：" + str(pred_item))
        print('实际：' + str(target_item) + '\n')
        image_loader = transforms.ToPILImage()
        image = image_data[i].clone()
        image = image.squeeze(0)
        image = image_loader(image)
        if not os.path.exists('./error/{}'.format(target_item)):
            os.mkdir('./error/{}'.format(target_item))
        image.save('./error/{}/{}_{}.png'.format(target_item, batch_idx, pred_item))

print('{}/{}， 识别率{:.2f}%'.format(correct, len(test_set), 100. * correct / len(test_set)))