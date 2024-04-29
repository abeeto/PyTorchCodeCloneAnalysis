import torch
import torchvision
from torchvision import transforms
from siamese_dataset import SiameseNetworkDataset
from siamese_network import SiameseNetwork
from helpers import imshow
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F


class Config():
    testing_dir = "/home/wingman2/datasets/personas/test/"
    # testing_dir = "/home/wingman2/code/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/faces/testing/"


model = SiameseNetwork().cuda()
model.load_state_dict(torch.load('/home/wingman2/models/siamese-faces-160.pt'))
model.eval()

data_transforms_test = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

folder_dataset_test = ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=data_transforms_test,
                                        should_invert=False)
test_dataloader = DataLoader(siamese_dataset, num_workers=8, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

for i in range(10):
    _, x1, label2 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = model(Variable(x0).cuda(), Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))