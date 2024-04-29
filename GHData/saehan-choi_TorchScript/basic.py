import torch.nn.functional as F
import torchvision.models as models
import torch
import time



def test1():
    class MyCell(torch.nn.Module):
        def __init__(self):
            super(MyCell, self).__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x, h):
            new_h = torch.tanh(self.linear(x) + h)
            return new_h, new_h

    my_cell = MyCell()
    x, h = torch.rand(3, 4), torch.rand(3, 4)
    st = time.time()
    print(my_cell(x, h))
    traced_cell = torch.jit.trace(my_cell, (x, h))
    print('\n')
    print(traced_cell(x, h))

    # (tensor([[ 0.5400,  0.8444, -0.3510, -0.2748],
    #         [-0.1721,  0.6323,  0.6192,  0.3357],
    #         [ 0.4561,  0.8881,  0.1714, -0.5267]], grad_fn=<TanhBackward0>), tensor([[ 0.5400,  0.8444, -0.3510, -0.2748],
    #         [-0.1721,  0.6323,  0.6192,  0.3357],
    #         [ 0.4561,  0.8881,  0.1714, -0.5267]], grad_fn=<TanhBackward0>))
    # 0.001999378204345703s passed


    # (tensor([[ 0.5400,  0.8444, -0.3510, -0.2748],
    #         [-0.1721,  0.6323,  0.6192,  0.3357],
    #         [ 0.4561,  0.8881,  0.1714, -0.5267]], grad_fn=<TanhBackward0>), tensor([[ 0.5400,  0.8444, -0.3510, -0.2748],
    #         [-0.1721,  0.6323,  0.6192,  0.3357],
    #         [ 0.4561,  0.8881,  0.1714, -0.5267]], grad_fn=<TanhBackward0>))
    # 0.0020046234130859375s passed

    # 둘의 시간차이는 크게 나지않았음

def test2():
    class MyDecisionGate(torch.nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x
            else:
                return -x

    class MyCell(torch.nn.Module):
        def __init__(self, dg):
            super(MyCell, self).__init__()
            self.dg = dg
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x, h):
            new_h = torch.tanh(self.dg(self.linear(x)) + h)
            return new_h, new_h

    my_cell = MyCell(MyDecisionGate())
    x, h = torch.rand(3, 4), torch.rand(3, 4)
    traced_cell = torch.jit.trace(my_cell, (x, h))

    print(traced_cell.dg.code)
    print(traced_cell.code)

    #   if x.sum() > 0:
    # def forward(self,
    #     argument_1: Tensor) -> NoneType:
    #   return None

    # def forward(self,
    #     x: Tensor,
    #     h: Tensor) -> Tuple[Tensor, Tensor]:
    #   dg = self.dg
    #   linear = self.linear
    #   _0 = (linear).forward(x, )
    #   _1 = (dg).forward(_0, )
    #   _2 = torch.tanh(torch.add(_0, h))
    #   return (_2, _2)
    # 여기에 if else같은 control 구문이 없는것을 알 수 있음

def test3():
    
    class MyDecisionGate(torch.nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x
            else:
                return -x

    class MyCell(torch.nn.Module):
        def __init__(self, dg):
            super(MyCell, self).__init__()
            self.dg = dg
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x, h):
            new_h = torch.tanh(self.dg(self.linear(x)) + h)
            return new_h, new_h

    scripted_gate = torch.jit.script(MyDecisionGate())

    my_cell = MyCell(scripted_gate)
    scripted_cell = torch.jit.script(my_cell)

    print(scripted_gate.code)
    print(scripted_cell.code)

def test4():

    r18 = models.resnet18(pretrained=True)       # 이제 사전 학습된 모델의 인스턴스가 있습니다.
    r18_scripted = torch.jit.script(r18)         # *** 여기가 TorchScript로 내보내는 부분입니다.
    dummy_input = torch.rand(1, 3, 224, 224)     # 빠르게 테스트 해봅니다.

    print(r18_scripted)
    time.time

    
if __name__ == '__main__':
    test4()





