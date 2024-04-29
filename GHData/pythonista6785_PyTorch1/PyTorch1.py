import sys
import numpy as np
import torch
from SimpleModel import SimpleModel
from MyDataSet import MyDataSet
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


def main():
    x = np.arange(100)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device ='cpu'

    # convert numpy to tensor
    x_tensor = torch.from_numpy(x).float().to(device)
    print(type(x_tensor))

    # convert tesnor back to numpy
    x_cpu = x_tensor.cpu().numpy()
    print(type(x_cpu))

    # requires_grad = True or False to make a variable trainable or not 
    w = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
    print('w =', w)

    # we can either create regular tenosrs and send them to the device 
    a = torch.randn(1, dtype=torch.float).to(device)
    b = torch.randn(1, dtype=torch.float).to(device)
    # and then set them as requireing gradient
    a.requires_grad_()
    b.requires_grad_(False)
    print(a)
    print(b)

    # we can specify the device at the moment of creation - RECOMMENDED!
    torch.manual_seed(42)
    a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    print(a)
    print(b)

    # methods that end with _ do inplace modification
    # loss.backward()  to compute gradients 
    # .zeros()  to  zero out gradients 
    # .grad attribute to examine the value of the gradient for a given tensor 

    w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

    lr = 1e-2
    n_epochs = 1000

    # define a MSE loss function
    x_train = np.arange(0, 10)
    y_train = x_train * 2 + 0.4   # y = 2x + 0.4 

    x_train_tensor  = torch.from_numpy(x_train).float().to(device)
    y_train_tensor  = torch.from_numpy(y_train).float().to(device)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    model = SimpleModel().to(device)
    model.train()  # set the model in train mode
    # optimizer = torch.optim.SGD([w, b], lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        #aout = w * x_train_tensor + b
        aout = model(x_train_tensor)
        loss = loss_fn(y_train_tensor, aout)

        loss.backward()         # compute gradients 
        optimizer.step()        # update weigths biases 
        optimizer.zero_grad()   # clear gradients 

    print('w =',w)
    print('b =',b)
    print(model.state_dict())   # model's weight and parameters 
    
    #----------------DataLoader Version-----------------
    x_train = np.arange(0,10)
    y_train = x_train * 2 + 0.4  # y = 2x + 0.4 
    # without.to(device)
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    mydataset = MyDataSet(x_train_tensor, y_train_tensor)
    train_dataset, val_dataset = random_split(mydataset, [8, 2])

    train_loader = DataLoader(dataset=train_dataset, batch_size=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=2)
    print(train_dataset[0])
    #train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)


    losses = []
    val_losses = []
    lr = 1e-2
    n_epochs = 100
    loss_fn = torch.nn.MSELoss(reduction='mean')
    model = SimpleModel().to(device)
    model.train() # set model in train mode
    # optimizer = torch.optim.SGD([w, b], lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            aout = model(x_batch)
            loss = loss_fn(y_batch, aout)

            loss.backward()        # compute gradients 
            optimizer.step()       # update weigths, biases
            optimizer.zero_grad()  # clear gradients 
            losses.append(loss)
        with torch.no_grad():   # turn of gradients calculation
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                model.eval()  # set model to evaluation mode 

                aout = model(x_val)
                val_loss = loss_fn(y_val, aout)
                val_losses.append(val_loss.item())
                print('epoch' + str(epoch) + ' Validation loss = ' + str(val_loss))
    print(model.state_dict())



if __name__ == "__main__":
    sys.exit(int(main() or 0))