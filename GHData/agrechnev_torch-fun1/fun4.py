# By Oleksiy Grechnyev 3/19/20

import onnx
import torch

torch.manual_seed(0)


class Yen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output*result

class Yen2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, C, S):
        result = (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) -
              C.unsqueeze(0).unsqueeze(0)).pow_(2).sum(3).mul_(S.view(1, 1, C.size(0)))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ScaledL2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, C, S):
        SL = (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) -
              C.unsqueeze(0).unsqueeze(0)).pow_(2).sum(3).mul_(S.view(1, 1, C.size(0)))
        ctx.save_for_backward(X, C, S, SL)
        return SL

    @staticmethod
    def backward(ctx, GSL):
        X, C, S, SL = ctx.saved_variables

        tmp = (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) - C.unsqueeze(0).unsqueeze(0)).mul_(
            (2 * GSL).mul_(S.view(1, 1, C.size(0))).unsqueeze(3)
        )

        GX = tmp.sum(2)
        GC = tmp.sum((0, 1)).mul_(-1)
        GS = SL.div(S.view(1, 1, C.size(0))).mul_(GSL).sum((0, 1))

        return GX, GC, GS

def scaled_l2_oleksiy(X, C, S):
        SL = (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) -
              C.unsqueeze(0).unsqueeze(0)).pow_(2).sum(3).mul_(S.view(1, 1, C.size(0)))
        return SL



if False:
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    # y = x.exp()
    y = Yen.apply(x)
    s = y.sum()
    s.backward()
    print(f'x = {x}')
    print(f'x.grad = {x.grad}')
    print(f'y = {y}')
    print(f's = {s}')


if False:
    x = torch.randn(3, 2, 1, requires_grad=True)
    c = torch.randn(4, 5)
    s = torch.randn(2, 2)

    # y = c.unsqueeze(0).unsqueeze(0)
    y = ScaledL2.apply(x, c, s)
    # y = scaled_l2_oleksiy(x, c, s)
    res = y.sum()
    res.backward()
    print(f'x.shape = {x.shape}')
    print(f'y.shape = {y.shape}')
    print(f'res = {res}')
    print(f'x.grad.sum() = {x.grad.sum()}')


class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.c = torch.randn(4, 5)
        self.s = torch.randn(2, 2)

    def forward(self, x):
        # return ScaledL2.apply(x, self.c, self.s)
        return scaled_l2_oleksiy(x, self.c, self.s)


if False:
    net = Net1()
    print(f'net = {net}')
    x = torch.randn(3, 2, 1, requires_grad=True)
    y = net(x)
    res = y.sum()
    res.backward()
    print(f'res = {res}')
    print(f'x.grad.sum() = {x.grad.sum()}')

if True:
    net = Net1()
    print(f'net = {net}')
    x = torch.randn(3, 2, 1, requires_grad=True)

    torch.onnx.export(net, x,
                      'fun.onnx',  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      verbose=True,
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output']  # the model's output names
                      )

    onnx_model = onnx.load('fun.onnx')
    print("Model successfully saved and loaded, checking ...")
    onnx.checker.check_model(onnx_model)