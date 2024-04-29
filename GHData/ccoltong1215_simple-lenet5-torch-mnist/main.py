
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset
from model import LeNet5, CustomMLP,LeNet5_regulized
import numpy as np
import matplotlib.pyplot as plt
import wandb


def train(model, trn_loader, device, criterion, optimizer,epoch,modelname):
    """ Train function
    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    model.to(device)
    model.train()
    trn_loss, acc = [], []
    for m in range(epoch):
        train_loss = 0
        trainacc = 0
        for i, (images, labels) in enumerate(trn_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss
            temp_acc = torch.mean(torch.eq(torch.argmax(outputs, dim=1), labels).to(dtype=torch.float64))
            trainacc += temp_acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (i) % 1000 == 0:
                print("\r {} Step [{}] Loss: {:.4f} acc: {:.4f}\nlabel".format(modelname,i, loss.item(), temp_acc),labels,"\n output", torch.argmax(outputs, dim=1))
        trainacc = trainacc / trn_loader.__len__()
        train_loss = train_loss / (trn_loader.__len__())     #10은 batchsize, 원래는 argument로 받아와서 사용가능
        print("{} training {} epoch  Loss: {:.4f} acc: {:.4f}".format(modelname,m, train_loss, trainacc))
        trn_loss.append(train_loss.item())
        acc.append(trainacc.item())
        epochlist = range(epoch)

        data = [[x, y] for (x, y) in zip( epochlist,acc)]
        data2 = [[x, y] for (x, y) in zip(epochlist, trn_loss)]
        table = wandb.Table(data=data, columns=[ "epoch","{}Acc".format(modelname)])
        table2 = wandb.Table(data=data2, columns=["epoch", "{}loss".format(modelname)])
        wandb.log({"{}Acc".format(modelname): wandb.plot.line(table, "epoch", "{}Acc".format(modelname),title=  "{}Acc graph".format(modelname))})
        wandb.log({"{}loss".format(modelname): wandb.plot.line(table2, "epoch", "{}loss".format(modelname),title="{}loss graph".format(modelname))})

    trn_loss = np.array(trn_loss)
    acc=np.array(acc)
    try:
        dummy_input = torch.randn(10,1,28,28,device=device)
        input_names = ["input_0"]
        output_names = ["output_0"]
        dummy_output = model(dummy_input)
        torch.onnx.export(model, dummy_input, "{}.onnx".format(modelname), verbose=True, input_names=input_names,output_names=output_names)
    except:
        pass
    return trn_loss, acc


def test(model, tst_loader, device, criterion,modelname):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    model.to(device)
    model.eval()
    tst_loss, acc = [],[]
    test_loss=0
    test_acc=0
    with torch.no_grad(): # 미분 안함,
        for i, (images, labels) in enumerate(tst_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss
            temp_acc = torch.mean(torch.eq(torch.argmax(outputs, dim=1), labels).to(dtype=torch.float64))
            test_acc += temp_acc
            tst_loss.append(loss.item())
            acc.append(temp_acc.item())
            if (i) % 100 == 0:
                print(" Step [{}] Loss: {:.4f} acc: {:.4f}".format(i, loss.item(), temp_acc))
                print("label", labels)
                print("output", torch.argmax(outputs, dim=1))

        test_acc = test_acc/tst_loader.__len__()
        test_loss = test_loss / (tst_loader.__len__())
        print("TEST Step [{}] Loss: {:.4f} acc: {:.4f}".format(tst_loader.__len__(), test_loss, test_acc))
    tst_loss=np.array(tst_loss).astype(float)
    acc=np.array(acc).astype(float)


    wandb.log({"{}Acc_test".format(modelname): test_acc,
               "{}loss_test".format(modelname): test_loss})



    return tst_loss, acc




# import some packages you need here

def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    wandb.init(project="simple_MNIST_report", config={
    })
    roottrain='data/train'
    roottest ='data/test'
    epoch = 100

    # declare pipeline
    trainloader = DataLoader(dataset=Dataset(root=roottrain,normalize=True),  #################################################
                         batch_size=10,
                         shuffle=True)
    trainloader_normalize = DataLoader(dataset=Dataset(root=roottrain,normalize=False),  #################################################
                         batch_size=10,
                         shuffle=True)
    testloader = DataLoader(dataset=Dataset(root=roottest,normalize=False),  ################################################
                        batch_size=10,
                        shuffle=False)
    device = torch.device("cuda:0")


    #declare model and opt and loss
    LeNet5_model = LeNet5()
    criterionLeNet = torch.nn.CrossEntropyLoss()
    optimizerLeNet = torch.optim.SGD(LeNet5_model.parameters(), lr=0.001, momentum=0.9)

    LeNet5_regulized_model = LeNet5_regulized()
    criterionLeNet_regulized = torch.nn.CrossEntropyLoss()
    optimizerLeNet_regulized = torch.optim.SGD(LeNet5_regulized_model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.001) ## L2 regularization

    CustomMLP_model = CustomMLP()
    criterionCustomMLP = torch.nn.CrossEntropyLoss()
    optimizerCustomMLP = torch.optim.SGD(CustomMLP_model.parameters(), lr=0.001, momentum=0.9)

    wandb.watch(
        LeNet5_model
        )
    wandb.watch(
        CustomMLP_model
        )
####################################################################################
    #start training
    lenet5_regulizedtrnloss, lenet5_regulizedtrnacc = train(model=LeNet5_regulized_model, trn_loader=trainloader_normalize, device=device, criterion=criterionLeNet_regulized,
                                        optimizer=optimizerLeNet_regulized,epoch=epoch,modelname="lenet_regulized")
    lenet5_regulizedtstloss, lenet5_regulizedtstacc = test(model=LeNet5_regulized_model, tst_loader=testloader, device=device, criterion=criterionLeNet_regulized,modelname="lenet_regulized")

    lenet5trnloss, lenet5trnacc = train(model=LeNet5_model, trn_loader=trainloader, device=device, criterion=criterionLeNet,
                                        optimizer=optimizerLeNet,epoch=epoch,modelname="lenet")
    lenet5tstloss, lenet5tstacc = test(model=LeNet5_model, tst_loader=testloader, device=device, criterion=criterionLeNet,modelname="lenet")

    CustomMLPtrnloss, CustomMLPtrnacc = train(model=CustomMLP_model, trn_loader=trainloader, device=device,
                                              criterion=criterionCustomMLP, optimizer=optimizerCustomMLP,epoch=epoch,modelname="custom")
    CustomMLPtstloss, CustomMLPtstacc = test(model=CustomMLP_model, tst_loader=testloader, device=device, criterion=criterionCustomMLP,modelname="custom")




    fig= plt.figure()

    lossplt=fig.add_subplot(2, 2, 1)

    plt.plot(range(epoch), lenet5trnloss, color='g', label='LeNet5 train loss')
    plt.plot(range(epoch), lenet5_regulizedtrnloss,color='r'   ,label='LeNet5_regulized train loss'    )
    plt.plot(range(epoch), CustomMLPtrnloss,color='b',label='Custom MLP train loss')
    plt.legend(loc='upper right',bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('epoch (x100)')
    plt.ylabel('loss')
    plt.title('Train Loss')

    accplt=fig.add_subplot(2, 2, 2)
    plt.plot(range(epoch), lenet5trnacc          ,color='g' ,label='LeNet5 train accuracy'    )
    plt.plot(range(epoch), lenet5_regulizedtrnacc, color='r',label='LeNet5_regulized train loss')
    plt.plot(range(epoch), CustomMLPtrnacc       ,color='b' ,label='Custom MLP train accuracy')
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('epoch (x100)')
    plt.ylabel('acc')
    plt.title('Train Accuracy')
    #
    lenetplt=fig.add_subplot(2, 2, 3)
    plt.plot(range(int((testloader.__len__()))),lenet5tstloss          , color='r', label='LeNet5 test loss')
    plt.plot(range(int((testloader .__len__()))) ,lenet5_regulizedtstloss,color='r',label='LeNet5 regulized test loss_'     )
    plt.plot(range(int((testloader .__len__()))) ,CustomMLPtstloss       ,color='m' ,label='Custom MLP test loss' )
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('epoch (x100)')
    plt.title('TEST Loss')
    #
    customplt=fig.add_subplot(2, 2, 4)
    plt.plot(range(int(testloader.__len__())),lenet5tstacc          , color='r', label='LeNet5 test accuracy')
    plt.plot(range(int(testloader .__len__())) ,lenet5_regulizedtstacc,color='r',label='LeNet5 regulized test accuracy '     )
    plt.plot(range(int(testloader .__len__())) ,CustomMLPtstacc       ,color='m' ,label='Custom MLP test accuracy' )
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('epoch (x100)')
    plt.title('TEST Acc')
    plt.show()
    plt.savefig('/fig.png')


if __name__ == '__main__':
    main()
    ### MNIST WEB app with python - Flask  http://hanwifi.iptime.org:9000/
    ### 19512062 young il han
    ### ccoltong1215@seoultech.ac.kr
    ### https://github.com/ccoltong1215/simple-lenet5-torch-mnist
    ### https://wandb.ai/ccoltong1215/simple_MNIST_report/runs/16etprdd?workspace=user-ccoltong1215