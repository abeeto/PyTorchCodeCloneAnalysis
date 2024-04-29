# 자주 쓸거 같은 기능들 모음
# collection of utilizable functions
import torch
# import torch.nn

# Print number of parameters of model
def numParms(model):
    '''
    input
        [nn.Module] : torch model
    output
        [int] : number of parameters
    '''
    print("# {} parameters:".format(model.__class__.__name__), 
        sum(param.numel() for param in model.parameters()))

# Print parameters name and size
def printStates(model):
    '''
    input
        [nn.Module] : torch model
    output
        None
    '''
    print("{}'s state_dict:".format(model.__class__.__name__))
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Save model's parameters as state_dict
def saveModel(model, PATH):
    '''
    input
        [nn.Module] : torch model
        [string] : path to save
    output
        None
    '''
    torch.save(model.state_dict(), PATH)

# Load model's parameters with satate_dict
def loadModel(model, PATH, train=False):
    '''
    input
        [nn.Module] : torch model
        [string] : path to load
        [bool] : determine .train() .eval()

    output : None
    '''
    model.load_state_dict(torch.load(PATH))
    if train:
        model.train()
    else:
        model.eval()



'''
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)

몇몇 키를 제외하고 state_dict 의 일부를 불러오거나,
적재하려는 모델보다 더 많은 키를 갖고 있는 state_dict 를 불러올 때에는 
load_state_dict() 함수에서 strict 인자를 False 로 설정하여 
일치하지 않는 키들을 무시하도록 해야 합니다.
한 계층에서 다른 계층으로 매개변수를 불러오고 싶지만, 
일부 키가 일치하지 않을 때에는 적재하려는 모델의 키와 일치하도록 
state_dict 의 매개변수 키의 이름을 변경하면 됩니다.
'''