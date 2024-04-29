import torch
import netron




modelData = "/home/rosetta/wsddn.pytorch/weight/visual1.pt"  # 定义模型数据保存的路径
netron.start(modelData)  # 输出网络结构
