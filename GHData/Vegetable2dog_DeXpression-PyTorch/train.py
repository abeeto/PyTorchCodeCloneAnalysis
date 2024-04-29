import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from Model import Block1, Block2, Block3

# defining hyper-parameters
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
TRAIN_DATA_PATH = "./train_set_2"

# creating training and test tensors
transform = transforms.Compose(
    [transforms.CenterCrop(480),
     transforms.Resize(224),
     transforms.Grayscale(3),
     transforms.RandomRotation(2),
     transforms.ToTensor()])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
classes = ('Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise',)

block1 = Block1()
block2 = Block2()
block3 = Block3()
net = nn.Sequential(block1, block2, block3)
criterion = nn.NLLLoss()  # we want to use NLLLoss over here
optimizer = optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.999), eps=0.1)
for epoch in range(EPOCHS):
    running_loss = 0.0
    running_loss1 = 0.0
    for i, data in enumerate(train_data_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss1 += loss.item()
        print('[%d, %5d]' % (epoch + 1, i + 1))
    print('EPOCH: %d, LOSS: %.5f' % (epoch + 1, running_loss1/2373.0))  # 18981/8
    running_loss1 = 0.0
    # saving after each epoch
    torch.save({'net_state_dict': net.state_dict(),
                }, 'last_model_state.pth')

print("Finished Training")
# saving the final model
torch.save({'net_state_dict': net.state_dict(),
                }, 'last_model_state.pth')
