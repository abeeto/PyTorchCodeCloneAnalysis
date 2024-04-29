import torch
from torch import nn

from model import PolicyFeedforward as PolicyFeed
from typing import Dict
import numpy as np

# 将learch的模型转化为libtorch支持加载的格式

class StandardScaler(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mean = nn.Parameter(torch.ones(size=(input_dim,)), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(size=(input_dim,)), requires_grad=True)

    def forward(self, x):
        return torch.div(torch.sub(x, self.mean), self.scale)


class PolicyFeedforward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.standardScaler = StandardScaler(input_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(self.standardScaler(x))


if __name__ == '__main__':
    model_path = "model/feedforward_3.pt"

    models = torch.load(model_path, map_location='cpu')
    state_dict: Dict = {"mean": torch.from_numpy(models["scaler"].mean_),
                        "scale": torch.from_numpy(models["scaler"].scale_)}
    model: PolicyFeedforward = PolicyFeedforward(47, 64)
    model.standardScaler.load_state_dict(state_dict, strict=True)
    model.load_state_dict(models["state_dict"], strict=False)
    # C++加载的模型必须用TorchScript保存，此处模型没有if-else控制刘，torch.jit.trace即可
    traced_script_module = torch.jit.trace(model, torch.ones(size=(1, 47)))
    traced_script_module.save("model/feedforward.pt")

    # 下面的代码用来判断新模型和旧模型输出是否一致
    model1: PolicyFeed = PolicyFeed.load(model_path)
    vec = np.ones(shape=(1, 47))
    vec1 = model1.scaler.transform(vec)
    result1 = model1(torch.from_numpy(vec1).float())
    result2 = model(torch.from_numpy(vec).float())

    print(result1)
    print(result2)