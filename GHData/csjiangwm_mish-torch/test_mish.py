import pytest
import torch
from mish import MemoryEfficientMish, Mish, MishImplementation
from torch.testing import assert_allclose

def get_input_params():
    devs = ['cpu']
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        devs += ['cuda:0'] # TODO: Allow other devices
    dev_types = [(dtype,device)
                 for dtype in [torch.float16,torch.float32,torch.float64]
                 for device in devs
                 # Basic ops not supported on CPU/Half, could test by converting but skip for now
                 if not (dtype==torch.float16 and torch.device(device).type == 'cpu')] 
    inputs = [(ndim,dtype,device)
              for (dtype,device) in dev_types
              for ndim in [1,2,3,4,8]]
    return inputs

@pytest.fixture(params=get_input_params())
def test_input(request):
    ndim,dtype,device = request.param
    sz = (2,) * (ndim-1) + (10,)
    if device == 'cpu' and dtype == torch.float16:
        return torch.randn(*sz).half() # No randn for half on CPU
    t = torch.randn(*sz, device=device, dtype=dtype)
    return t

def test_function(test_input):
    x1,x2 = (test_input.clone().requires_grad_() for i in range(2))
    m1 = Mish()
    y1 = m1(x1)
    l1 = y1.mean()
    exp, = torch.autograd.grad(l1, x1)

    y2 = MishImplementation.apply(x2)
    l2 = y2.mean()
    res, = torch.autograd.grad(l2, x2)
    assert_allclose(res, exp)

def test_module(test_input):
    x1,x2 = (test_input.clone().requires_grad_() for i in range(2))

    m1 = Mish()
    y1 = m1(x1)
    l1 = y1.mean()
    exp, = torch.autograd.grad(l1, x1)

    m2 = MemoryEfficientMish()
    y2 = m2(x2)
    l2 = y2.mean()
    res, = torch.autograd.grad(l2, x2)
    assert_allclose(y1, y2)
    assert_allclose(res, exp)

def test_gradient():
    inp = torch.randn(10, 10, dtype=torch.float64, requires_grad=True, device='cuda:0')
    assert torch.autograd.gradcheck(MishImplementation.apply, inp)

def test_gradgrad():
    inp = torch.randn(10, 10, dtype=torch.float64, requires_grad=True, device='cuda:0')
    assert torch.autograd.gradgradcheck(MishImplementation.apply, inp)


if __name__ == "__main__":
    pytest.main()