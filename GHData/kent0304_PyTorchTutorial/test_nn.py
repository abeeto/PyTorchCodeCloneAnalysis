import torch
import torch.nn as nn

input_features = 2
output_neurons = 1

activation = torch.nn.Tanh()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # layer1 という名のレイヤーを作る
        self.layer1 = nn.Linear(input_features, output_neurons)

    def forward(self, input):
        output = activation(self.layer1(input))
        return output

model = NeuralNetwork()

weight_array = nn.Parameter(torch.tensor([[0.6, -0.2]]))
bias_array = nn.Parameter(torch.tensor([0.8]))

model.layer1.weight = weight_array
model.layer1.bias = bias_array

# 重みやバイアスなどパラメータ情報取得
params = model.state_dict()
print(params)

X_data = torch.tensor([[1.0, 2.0]])
y_pred = model(X_data)

print(y_pred)
