import torch
from torch import nn, optim
from mnist_model import Model
from dataset_loader import ImagesLoader

training_paths = []

#Append all training folders
for i in range(10):
    training_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/{}/".format(i))

#Weight and bias initialization
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        print('Initializing conv2d weight and bias ')
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)

if __name__ == '__main__':
    #Load dataset
    loader = ImagesLoader(128)
    dataset = loader.get_dataset(training_paths, training=True)

    #Load model
    model = Model()
    #Apply weight and bias initialization
    model.apply(weight_init)

    #Set model device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Defíne loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    lossum,epoch= 0, 50

    print("Training started...")
    #Epoch loop
    for i in range(epoch):
        #Batch loop
        for images, labels in (zip(*dataset)):
            #Put images and labels to model device
            images = images.to(model.device)
            labels = labels.to(model.device)

            y = model(images)

            loss = criterion(y, labels)
            lossnumber = float(loss.data.cpu().numpy())
            lossum += lossnumber

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Average loss for epoch ",i+1, " - ",lossum / len(dataset[0]))
        lossum = 0


    #Save model
    PATH = './MNIST-MY.pth'
    torch.save(model.state_dict(), PATH)
    print("Model saved")
