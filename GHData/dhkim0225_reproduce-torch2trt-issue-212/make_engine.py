import torch
import torch.nn as nn
from torch2trt import torch2trt


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.expander = nn.Conv2d(3, 192, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.expander(x)
        x = self.upsample(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda:0')

    with torch.no_grad():
        sample_input = torch.rand(1, 3, 128, 128).to(device)
        model = TinyModel().to(device)
        model.eval()
        model_trt = torch2trt(model, [sample_input],
                              input_names=['input_image:0'],
                              output_names=['output:0'],
                              max_batch_size=1,
                              max_workspace_size=(1 << 32),
                              strict_type_constraints=False,
                              keep_network=True)

    torch.save(model_trt.state_dict(), '/home/trt_model.pth')
    with open('/home/model/my_model/1/model.plan', 'wb') as f:
        f.write(model_trt.engine.serialize())
