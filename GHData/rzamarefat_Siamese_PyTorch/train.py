import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import random
from PIL import Image
import matplotlib.pyplot as plt
from config import  config
from datasets import CustomDataset

train_image_folder = ImageFolder(config["training_path"])
test_image_folder = ImageFolder(config["testing_path"])

train_ds = CustomDataset(train_image_folder)
test_ds = CustomDataset(test_image_folder)

train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=64)
test_dataloader = DataLoader(test_ds, shuffle=True, batch_size=64)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    siamese_model = SiameseNetwork()
    siamese_model.to(device)
    criterion = ContrastiveLoss()
    optimizer = Adam(siamese_model.parameters(), lr=config['lr'])

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0,100):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)

            optimizer.zero_grad()
            output1,output2 = siamese_model(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())          

    # plt.plot(counter,loss_history)