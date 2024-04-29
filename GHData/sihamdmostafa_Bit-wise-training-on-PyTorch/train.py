
from models import *
import copy
import torchvision.datasets as ds
import torchvision.transforms as transforms

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_dataset(type="MNIST", batch_size=64):

    train_tf = [transforms.ToTensor()]
    test_tf = [transforms.ToTensor()]
    if type == "CIFAR10":

        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_tf = [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),   # some data augmentation
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(*stats, inplace=True)]
        test_tf = [transforms.ToTensor(), transforms.Normalize(*stats)]

    train_tfms = transforms.Compose(train_tf)
    test_tfms = transforms.Compose(test_tf)

    if type == "MNIST":
        train = ds.MNIST(root='MNIST/',
                               train=True,
                               transform=train_tfms,
                               download=True)

        test = ds.MNIST(root='MNIST/',
                              train=False,
                              transform=test_tfms,
                              download=True)
    elif type == "CIFAR10":

        train = ds.CIFAR10(root='CIFAR10/',
                         train=True,
                         transform=train_tfms,
                         download=True)

        test = ds.CIFAR10(root='CIFAR10/',
                        train=False,
                        transform=test_tfms,
                        download=True)
    else:
        raise ValueError("only valid options are MNIST or CIFAR10")

    # create data loader for training and test
    data_loader = torch.utils.data.DataLoader(dataset=train,
                                              batch_size=batch_size,
                                              shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(dataset=test,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    # load to device
    data_loader = DeviceDataLoader(data_loader, device)
    data_loader_test = DeviceDataLoader(data_loader_test, device)

    return data_loader, data_loader_test



def fit_net(model, train_loader, val_loader, config,grad_clip=None):
    '''
    training function according to config
    '''
    torch.cuda.empty_cache()
    history = []

    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=config["max_lr"])
    elif config["optimizer"] =='SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=config["max_lr"],momentum=0.9)
    else:
        raise ValueError("only valid options are Adam and SGD")


    sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["milestones"], gamma=0.1)

    # other schedulars if you want to expirement
    # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

    # check for saved checkpoints ( has to be entered manually with saved = (last epoch, saved weights)
    if config["saved"] == None:
        start = 0
    else:
        start = config["saved"][0]
        model.load_state_dict(torch.load(config["saved"][1]))

    best = 0

    for epoch in range(start, config["epochs"]):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping ( not used in bitwise training )
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

        # Record & update learning rate
        lrs.append(get_lr(optimizer))
        sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs

        model.epoch_end(epoch, result)
        history.append(result)
        # for each epoch save model if path exists
        if config["path"] != None:
            torch.save(model.state_dict(),
                       config["path"] + "/current_epoch_" + str(epoch) + "_" + str(history[-1]['val_acc']) + ".pth")
        # keep track of best model
        if history[-1]['val_acc'] > best:
            best = history[-1]['val_acc']
            best_epoch = epoch
            best_model = copy.deepcopy(model)
    if config["path"] != None:
        torch.save(best_model.state_dict(), config["path"] + "/best_epoch_" + str(best_epoch) + "_" + str(best) + ".pth")

    return history


import itertools
import os


def train_bit_list(config,nbits,model_type="LeNet", configs=None):
    # in case no list is specified we take all the combinition ( Note that would be heavy on the hardware )
    if configs == None:
        configs = list(map(list, itertools.product([0, 1], repeat=nbits)))[1:]
    nbits = len(configs[0])
    acc = []
    if config["path"] == None:
        path = ""
    else:
        path = config["path"]

    for trainableBits in configs:

        save_path =  path + "/"+model_type+"/{}_bits/{}".format(nbits, trainableBits)
        os.makedirs(save_path, exist_ok=True)

        config["trainable_bits"] = trainableBits
        config["nbBits"] = nbits
        config["inference_sequence"] = [0, nbits - 1]
        config["path"] = save_path



        if model_type == 'LeNet':
            model = to_device(LeNet(config), device)
            data_loader, data_loader_test = get_dataset("MNIST")
        elif model_type ==  'ResNet':
            model = to_device(resnet20(config), device)
            data_loader, data_loader_test = get_dataset("CIFAR10")
        elif model_type == "Conv6":
            model = to_device(Conv6(config), device)
            data_loader, data_loader_test = get_dataset("CIFAR10")
        elif model_type == "VGG":
            model = to_device(VGG("VGG11",config), device)
            data_loader, data_loader_test = get_dataset("CIFAR10")
        elif model_type == "EfficientNet":
            model = to_device(EfficientNetB0(config), device)
            data_loader, data_loader_test = get_dataset("CIFAR10")
        else:
            raise ValueError("only valid options are LeNet, ResNet, Conv6, VGG, EfficientNet")
        print("Traning for {} config : ".format(trainableBits))

        history = []
        history += fit_net(model, data_loader, data_loader_test,config)

        acc.append(np.array([h["val_acc"] for h in history]).max())
    return configs, acc


def LeNet_train(config=None):

    nbits = 2
    trainableBits = [0, 1]


    if config == None:

        config = {
            "default": False,  # is this is true then it trains in the standard way, using default float32 weights
            "nbBits": len(trainableBits),  # bit-depth used for weights
            "trainable_bits": trainableBits,  # specify which bits are trainable, kind related to the previous
            "inference_sequence": [0, nbits - 1],  # this allows us to choose which bits participate in the calculation
            "epochs": 30,  # number of epochs
            "max_lr": 0.006,  # , learning rate
            "optimizer": "Adam",  # type of optimizer
            "milestones": [20, 25],  # epochs where to decrese the learning rate by 0.1 factor
            "path": None,  # where to save the execution
            "saved": None  # continue from checkpoint
        }

    data_loader, data_loader_test = get_dataset("MNIST")
    model = to_device(LeNet(config), device)
    history = []
    history += fit_net(model, data_loader, data_loader_test,config)


def ResNet_train(config=None):

    nbits = 2
    trainableBits = [0, 1]


    if config == None:

        config = {
            "default": False,  # is this is true then it trains in the standard way, using default float32 weights
            "nbBits": len(trainableBits),  # bit-depth used for weights
            "trainable_bits": trainableBits,  # specify which bits are trainable, kind related to the previous
            "inference_sequence": [0, nbits - 1],  # this allows us to choose which bits participate in the calculation
            "epochs": 60,  # number of epochs
            "max_lr": 0.006,  # , learning rate
            "optimizer": "Adam",  # type of optimizer
            "milestones": [45, 55],  # epochs where to decrese the learning rate by 0.1 factor
            "path": None,  # where to save the execution
            "saved": None  # continue from checkpoint
        }

    data_loader, data_loader_test = get_dataset("CIFAR10")
    model = to_device(resnet20(config), device)
    history = []
    history += fit_net(model, data_loader, data_loader_test,config)


def Conv6_train(config=None):

    nbits = 2
    trainableBits = [0, 1]


    if config == None:

        config = {
            "default": False,  # is this is true then it trains in the standard way, using default float32 weights
            "nbBits": len(trainableBits),  # bit-depth used for weights
            "trainable_bits": trainableBits,  # specify which bits are trainable, kind related to the previous
            "inference_sequence": [0, nbits - 1],  # this allows us to choose which bits participate in the calculation
            "epochs": 40,  # number of epochs
            "max_lr": 0.006,  # , learning rate
            "optimizer": "Adam",  # type of optimizer
            "milestones": [25, 35],  # epochs where to decrese the learning rate by 0.1 factor
            "path": None,  # where to save the execution
            "saved": None  # continue from checkpoint
        }

    data_loader, data_loader_test = get_dataset("CIFAR10")
    model = to_device(Conv6(config), device)
    history = []
    history += fit_net(model, data_loader, data_loader_test,config)


def VGG_train(config=None):

    nbits = 2
    trainableBits = [0, 1]


    if config == None:

        config = {
            "default": False,  # is this is true then it trains in the standard way, using default float32 weights
            "nbBits": len(trainableBits),  # bit-depth used for weights
            "trainable_bits": trainableBits,  # specify which bits are trainable, kind related to the previous
            "inference_sequence": [0, nbits - 1],  # this allows us to choose which bits participate in the calculation
            "epochs": 40,  # number of epochs
            "max_lr": 0.006,  # , learning rate
            "optimizer": "Adam",  # type of optimizer
            "milestones": [25, 35],  # epochs where to decrese the learning rate by 0.1 factor
            "path": None,  # where to save the execution
            "saved": None  # continue from checkpoint
        }

    data_loader, data_loader_test = get_dataset("CIFAR10")
    model = to_device(VGG("VGG11",config), device)
    history = []
    history += fit_net(model, data_loader, data_loader_test,config)


def EfficientNet_train(config=None):

    nbits = 2
    trainableBits = [0, 1]


    if config == None:

        config = {
            "default": False,  # is this is true then it trains in the standard way, using default float32 weights
            "nbBits": len(trainableBits),  # bit-depth used for weights
            "trainable_bits": trainableBits,  # specify which bits are trainable, kind related to the previous
            "inference_sequence": [0, nbits - 1],  # this allows us to choose which bits participate in the calculation
            "epochs": 40,  # number of epochs
            "max_lr": 0.006,  # , learning rate
            "optimizer": "Adam",  # type of optimizer
            "milestones": [25, 35],  # epochs where to decrese the learning rate by 0.1 factor
            "path": None,  # where to save the execution
            "saved": None  # continue from checkpoint
        }

    data_loader, data_loader_test = get_dataset("CIFAR10")
    model = to_device(EfficientNetB0(config), device)
    history = []
    history += fit_net(model, data_loader, data_loader_test,config)



if __name__ == '__main__':
    LeNet_train()
