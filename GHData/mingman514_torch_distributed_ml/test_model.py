import ddp_benchmark
from ddp_benchmark import *
from torchsummary import summary
import torch as th 



def get_test_dataset():
    transform = transforms.Compose(
            [transforms.ToTensor(),
#            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            transforms.Normalize((0.1307,), (0.3081,))])

    dataset = datasets.MNIST(
#    dataset = datasets.CIFAR10(
        datapath,
        train=False,
        download=True,
        transform=transform)

    test_set = torch.utils.data.DataLoader(dataset, batch_size=64)                                    
    return test_set



if __name__ == '__main__':
    model = ddp_benchmark.Net()
    with open("distribute_learned.pkl", "rb") as f:
        state_dict = torch.load(f)
    
    new_state_dict = {} 
    for key in state_dict.keys():
        if key[:7] == 'module.':
            k = key[7:] 
        else:
            k = key 
        new_state_dict[k] = state_dict[key]
            
    model.load_state_dict(new_state_dict)
    test_dataset = get_test_dataset()
    test_accuracy(model, test_dataset) 
    
