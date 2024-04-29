import torch.nn as nn
from torchsummary import summary
# from models.base_model import BaseModel
from torchvision_my import models as BaseModel
# model_name ="mobilenetv3"
num_classes = 3
device ='cpu'

# model = BaseModel(name=model_name, num_classes=num_classes).to(device)
model = BaseModel.resnet18(num_classes=num_classes).to(device)
# self.model.load_state_dict(torch.load(args.model_weight_path, map_location=self.device))
# self.model.eval()


#输出每层网络参数信息
# summary(model,(3,224,224),batch_size=1,device="cpu")

summary(model,(3,256,256),batch_size=1,device=device)


