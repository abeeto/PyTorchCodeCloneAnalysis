from torchvision import models
import torch
# from torchvision_my.models.base_model_my import BaseModel
from torchvision_my import models as BaseModel
#加载model，model是自己定义好的模型
# resnet50 = models.resnet50(pretrained=True) 
resnet18 = models.resnet18(pretrained=True) 
if True:
    num_classes=3
    model =BaseModel.resnet18(num_classes=num_classes)
    
    #读取参数 预训练参数和当前网络参数
    # pretrained_dict =resnet50.state_dict() 
    pretrained_dict =resnet18.state_dict() 
    model_dict = model.state_dict() 
    
    #将pretrained_dict里不属于model_dict的键剔除掉 
    # pretrained_temp =  {k: v for k, v in pretrained_dict.items() if k in model_dict}     #这里其实也就踢出了后面不同的，全链接部分，最后保存的模型也就适合各种不同 num_classes
    method =1
    if method ==1:
        pretrained_temp=dict()
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if k =="fc.weight" or k == "fc.bias":
                    continue
                pretrained_temp[k]=v
    elif method ==2:
        pretrained_dict.pop("fc.weight")
        pretrained_dict.pop("fc.bias")
        pretrained_temp = pretrained_dict
    elif method ==3:
        pretrained_dict.popitem()
        pretrained_dict.popitem()
        pretrained_temp = pretrained_dict

    # 更新现有的model_dict 
    model_dict.update(pretrained_temp) 
    # 加载我们真正需要的state_dict 
    model.load_state_dict(model_dict)  

    for value1 ,value2 in zip(pretrained_temp.items(), model.state_dict().items()):
        print(value1,value2)  #对比打印最后值是否相等
    ###看来要保存通用模型,需要手动把不同地方去掉,###

    savePath = r"Z:\code\office-parameter\resnet\resnet_general_1.pth"
    torch.save(model.state_dict(),savePath)  
    print("ok!")

#导入
model_name ="resnet18"
num_classes=3
savePath = r"Z:\code\office-parameter\resnet\resnet_general_1.pth"
# model =BaseModel(model_name,num_classes)
model =BaseModel.resnet18(num_classes=num_classes)

model.load_state_dict(torch.load(savePath))
print("import ok!")

#save
# #只保存参数
# torch.save(resnet50.state_dict(),'ckp/model.pth')  
# #先导入模型结构，再调用保存的参数
# resnet=resnet50(pretrained=True)
# resnet.load_state_dict(torch.load('ckp/model.pth'))

# #保存 全部
# torch.save (model, PATH)
# #恢复
# model = torch.load(PATH)

#查看网络
# for key ,value in model.state_dict.items():
#     print(key,value)

#其他问题
# 1.删除键的头部
# 键是conv1.weight，预训练的键是module.conv1.weight，导致不匹配。所以下面的代码是让module. 去掉
# pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict2.items()}

# 2.补齐键的头部
# checkpoint={'module.'+k:v for k,v in pretrained_dict.items()}

#查看参数是否更新成功
for value1 ,value2 in zip(resnet18.state_dict().items(), model.state_dict().items()):
    print(value1,value2)  #对比打印最后值是否相等

print("ok para")