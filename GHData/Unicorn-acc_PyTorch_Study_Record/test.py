import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "11.jpg"
image = Image.open(image_path)
# print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # CIFA10 model结构
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.ReLU(),
            nn.MaxPool2d(2), nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1, 2), nn.ReLU(),
            nn.MaxPool2d(2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(),
            nn.MaxPool2d(2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("model_9", map_location=torch.device('cpu'))
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
model.eval()  # 评估模式
with torch.no_grad():  # 测试不用计算梯度
    output = model(image)

print(output)
print(output.argmax(1))
