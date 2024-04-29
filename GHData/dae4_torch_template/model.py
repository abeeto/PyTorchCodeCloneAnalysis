#%%
import timm
from torch import nn

available_models = timm.list_models()

class Model(nn.Module):
    """
    Thi is example code.
    Customize your self.
    """
    def __init__(self, base_model_name="alexnet", num_classes=10,freeze=False):
        super(Model, self).__init__()

        assert base_model_name in available_models, f"Please Check available pretrained model list at https://rwightman.github.io/pytorch-image-models/results/"
        
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        # self.freeze = freeze
        
        self.model = timm.create_model(self.base_model_name, pretrained=True, num_classes=num_classes)
        
        if freeze: 
            for layer_name, ops in self.model.named_children():
                if layer_name != 'classifier':
                    ops.requires_grad = False
    
        # self.init_params()
        
    def forward(self, x):
        x = self.model(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    