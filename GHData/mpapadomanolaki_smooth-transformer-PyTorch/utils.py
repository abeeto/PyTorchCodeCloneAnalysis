import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = 0
def to_cuda(v):
    if USE_CUDA:
        return v.cuda(DEVICE)
    return v

def initialize_model(model):
    torch.nn.init.kaiming_normal_(model.conv1.weight)
    torch.nn.init.kaiming_normal_(model.conv2.weight)
    torch.nn.init.kaiming_normal_(model.conv3.weight)
    torch.nn.init.kaiming_normal_(model.Up_conv4.weight)
    torch.nn.init.kaiming_normal_(model.Up_conv3.weight)
    torch.nn.init.kaiming_normal_(model.Up_conv2.weight)

    torch.nn.init.zeros_(model.affine_dense.weight)
    torch.nn.init.zeros_(model.affine_dense.bias)
    torch.nn.init.zeros_(model.deformable_layer.weight)
    torch.nn.init.zeros_(model.deformable_layer.bias)

    return model
