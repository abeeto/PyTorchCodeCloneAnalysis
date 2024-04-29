from turtle import forward
import torch
import torch.nn as nn
import os
os.system('cls')

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


#! inja model ro sakhtim
model = Model(n_input_features=3)

#! hala biaim parameter hasho bbinim chian:
print(f'model: {model}')
print(f'model_state_dict : {model.state_dict()}')
for i, param in enumerate(model.parameters()):
    print(''*2)
    print(f'param {i} : \n')
    print(f'parameter shape : {param.shape}')
    print(f'parameter value : {param}')

print('\n\n\n\n')

#! hala aval khode model ro save mikonim baad state dict ro:
model_save_path = 'model.pth'
torch.save(model, model_save_path)

#! inja state_dict model ro save mikonim:
model_state_save_path = 'model_state.pth'
torch.save(model.state_dict(), model_state_save_path)

#! ye instance dige az model misazim : 
model2 = Model(n_input_features=3)

#TODO inja loadesh mikonim:
loaded_model = torch.load(model_save_path)
loaded_model.eval()
print(f'Loaded Model : {loaded_model.parameters()}\n\n')
for i, param in enumerate(loaded_model.parameters()):
    print(''*2)
    print(f'param {i} : \n')
    print(f'parameter shape : {param.shape}')
    print(f'parameter value : {param}')

print('\n\n\n\n')

#TODO inja state esho load mikonim:
device = torch.device('cuda')
loaded_state_model = Model(n_input_features=3).to(device=device)
loaded_state_model.load_state_dict(torch.load(model_state_save_path))
print(f'loaded state model : {loaded_state_model}\n\n')