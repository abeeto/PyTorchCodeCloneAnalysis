import torch
import pandas as pd

# from PyTorchRefresher.ann import Model


model = torch.load("./PyTorchRefresher/models/iris_model_complete.pt")
print(model.parameters)

data = pd.read_csv("./PyTorchRefresher/data/iris.csv")
features = data.drop(["target"], axis=1).values
label = data.target.values
sample = features[0]
sample_y = label[0]
sample = torch.tensor(sample, dtype=torch.float32)
model.eval()
with torch.no_grad():
    pred = model.forward(sample)
    print(pred)
    print(pred.argmax())
    print(sample_y)
