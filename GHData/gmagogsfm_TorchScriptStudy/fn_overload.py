import torch

@torch.jit._overload
def fn(x: int) -> int:
    assert False
    return x + 3


@torch.jit._overload
def fn(x: str) -> str:
    return x


def user(x: int):
    print(fn(x))
    print(fn(str(x)))

user(2)
print("==============")

scripted = torch.jit.script(user)
scripted(2)
