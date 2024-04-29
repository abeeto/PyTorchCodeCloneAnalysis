import io
import torch
import torch.nn as nn
import torch.nn.functional as F


# モデル定義
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.fc1 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.fc1(x))
        return x


# モデル初期化
print("1. モデル初期化")
model = TheModelClass()
print(model, "\n")

# モデルのバイナリストリーム化
print("2. モデルのバイナリストリーム化")
out_buf = io.BytesIO()
torch.save(model, out_buf)
print(out_buf.getvalue(), "\n")

# モデルの復元
print("3. モデルの復元")
in_model = torch.load(io.BytesIO(out_buf.getvalue()))
print(in_model, "\n")

# モデル Value 参照
print("4. モデル Value 参照")
print(in_model.state_dict(), "\n")
