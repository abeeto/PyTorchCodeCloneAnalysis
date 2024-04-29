import torch as t

if t.cuda.is_available():
    x = t.Tensor([5.5, 3])
    y = t.Tensor([5.5, 3])
    x = x.cuda()
    y = y.cuda()
    print(x + y)

x = t.randn(1)
if t.cuda.is_available():
    device = t.device("cuda")
    y = t.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(x.to(device))
    print(y.to("cpu"))
    print(z.to("cpu"))
