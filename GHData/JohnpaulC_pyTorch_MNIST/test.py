import torch
import cv2 as cv
from model import ConvNN
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

batch_size = 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ConvNN()
model.load_state_dict(torch.load('model.pt', map_location=device))
transform = transforms.Compose([
    transforms.ToTensor()
])
testset = MNIST('MNIST', train = False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)



Debug_flag = 0
with torch.no_grad():
    for j, test_data in enumerate(testloader):
        if Debug_flag == 3:
            break
        test_img = test_data[0]
        label = test_data[1]
        print(test_img.shape)
        _, pred = model(test_img, apply_softmax=True).max(dim=1)

        print(pred)
        pred = pred.numpy()
        plt.figure()
        for i in range(batch_size):
            plt.subplot(2, 2, i + 1).set_title("This is " + str(pred[i]))
            plt.axis('off')
            plt.imshow(test_img[i, ...].squeeze().numpy(), 'Greys_r')
        plt.show()

        Debug_flag += 1