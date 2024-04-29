import torch
import torchvision
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from siamese_dataset import SiameseNetworkDataset
import matplotlib.pyplot as plt
from siamese_network import SiameseNetwork
from contrastive_loss import ContrastiveLoss
from helpers import show_plot, imshow
from torch.utils.tensorboard import SummaryWriter


# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('/home/wingman2/runs/face_siamese_experiment_2')


class Config():
    training_dir = "/home/wingman2/datasets/personas/train/"
    # training_dir = "/home/wingman2/code/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/faces/training/"
    train_batch_size = 64
    train_number_epochs = 200


data_transforms_train = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

folder_dataset = ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=data_transforms_train,
                                        should_invert=False)

# TODO - un-comment this section to visualize train dataset examples
'''
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)
dataiter = iter(vis_dataloader)
example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())
'''

train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=8,
                              batch_size=Config.train_batch_size)

# Define a model
net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0

for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        writer.add_scalar("Contrastive loss/train", loss_contrastive, epoch)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
    if epoch > 0 and epoch % 20 == 0:
        torch.save(net.state_dict(), '/home/wingman2/models/siamese-faces-' + str(epoch) + ".pt")
        print("Model saved after epoch: " + str(epoch))

# Save final model
torch.save(net.state_dict(), '/home/wingman2/models/siamese-faces-final.pt')
writer.close()

show_plot(counter, loss_history)
