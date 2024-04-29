import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt

def change_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def train(net, device, train_loader, val_loader, cyclic = False, epochs = 30, lr = 0.001):
    '''
    Parameters
    ----------
    net : PyTorch Model
    device : device type, cpu or cuda
    train_loader : train data loaders
    val_loader : validation data loader
    epochs : # of epochs to run The default is 30.
    lr : Learning Rate The default is 0.001.

    Returns
    -------
    net : Model after training using given parameters

    '''
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay=1e-4, betas=(0.9, 0.999))
    num_batches = len(train_loader)
    scheduler = None
    print("Cyclic : ", cyclic)

    if cyclic :
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr = 4e-5,
                                                      max_lr = 4e-4,
                                                      step_size_up = num_batches,
                                                      cycle_momentum = False)
    else :
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               patience = 2, 
                                                               factor = 0.5, 
                                                               verbose = True)
    
    train_loss_arr = []
    val_loss_arr = []
    best_loss = 10.0
    # epochs = 30
    print("Number of batches = %s, lr = %s, epochs = %s"
          %(num_batches, get_lr(optimizer), epochs))
    print("Training Started at ", time.strftime("%H:%M:%S", time.localtime()))
    start = time.time()
    for epoch in range(0,epochs):  # loop over the dataset multiple times
        epoch_start = time.time()
        running_loss = 0.0
        epoch_loss = 0.0
        i = 0
        net.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            if inputs.shape[0] < 5:
                continue
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if cyclic :
                scheduler.step()
    
            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i != 0 and (i+1) % 300 == 0:    # print every 300 mini-batches
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i+1, running_loss / 300))
                running_loss = 0.0
        
        train_loss = epoch_loss/num_batches
        val_acc,val_top5, val_loss = val(net, device, val_loader, criterion)
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        
        if not(cyclic) :
            scheduler.step(val_loss)
            
        print("Epoch %s complete => Train_Loss : %.6f, Val_Loss : %.6f, Val_acc : %.2f , Val_top5_acc : %.2f , time taken : %s"%(epoch+1, train_loss, val_loss, val_acc, val_top5, time.time() - epoch_start))
        print("lr = %s"%(get_lr(optimizer)))
        #SAVE Best Model
        if epoch > (epochs/3) and val_loss < best_loss :
            best_loss = val_loss
            torch.save(net.state_dict(), 'SKNET.pt')
            print("Model saved")
    
    print("Training Finieshed at ", time.strftime("%H:%M:%S", time.localtime()))
    print("Time to train = %s seconds"%(time.time() - start))
    plt.plot(range(1, len(train_loss_arr)+1), train_loss_arr)
    plt.plot(range(1, len(val_loss_arr)+1), val_loss_arr)
    plt.legend(["train","val"])
    plt.savefig('train_graph.png')
    
    return net

def val(net, device, test_loader, criterion):
    correct = 0
    total = 0
    net.eval()
    loss = 0
    top5_correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            outputs = F.softmax(outputs, dim = 1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top5_correct += top5(outputs, labels)
            loss += criterion(outputs, labels)
    
    accuracy = 100 * correct / total
    top5_acc = 100* top5_correct / total
    loss = loss / len(test_loader)
    
    return accuracy, top5_acc, loss.item()

def test(net, device, test_loader):
    correct = 0.0
    total = 0.0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)   
            outputs = F.softmax(outputs, dim = 1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %s %%' % (
        100 * correct / total))
    
def top5(pred, y):
    k_vals = torch.topk(pred, 5).indices
    correct = 0
    for i in range(y.shape[0]):
        if y[i].item() in k_vals[i]:
            correct += 1
    return correct

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']