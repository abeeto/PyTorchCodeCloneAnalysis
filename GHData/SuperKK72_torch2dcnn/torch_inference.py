import torch
from torch_classes import LPRNet
import cv2
#初始化模型
net = LPRNet.LPRNet(8, False, 68, 0)
net_path = "./torch_models/LPRNet/weights/Final_LPRNet_model.pth"
net.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))
torch.save(net,"./torch_models_new/LPRNet.pth")

#读取输入
image_path = "./torch_images/dog.jpg"
image=cv2.imread(image_path, -1)
imgInfo = image.shape
n = 1
c = 3
h = 24
w = 94
image = cv2.resize(image,(h, w))
image = image.reshape(n, c, h, w)
image = torch.Tensor(image)

#运行网络inference
output = net.forward(image)
# print(output.shape)
outputArray = output.detach().numpy().flatten()#去除梯度信息

#保存前向结果
print("------------------------------------SAVE RESULT-------------------------------")
savePath = "./torch_inference_result/LPRNet_torch_inference_result"
outputFile = open(savePath, 'w')
for val in outputArray:
    outputFile.write(str(val))
    outputFile.write('\n')
print("---------------------------------------END------------------------------------")












