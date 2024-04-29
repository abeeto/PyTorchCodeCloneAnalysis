import torch


class TestInterface(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x: int):
        pass


class TestModule(torch.nn.Module):
    def __init__(self, v: int):
        super().__init__()
        self.x = v

    def forward(self, x: int):
        return self.x + x


def fn(m: TestInterface):
    return m(1)


m = torch.jit.script(TestModule(2))
print(fn(m))
