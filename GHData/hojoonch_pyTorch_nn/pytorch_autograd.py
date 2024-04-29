import torch
import math

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요.

def test_user_grad():
    # N은 배치 크기이며, D_in은 입력의 차원입니다;
    # H는 은닉층의 차원이며, D_out은 출력 차원입니다.
    #N, D_in, H, D_out = 64, 1000, 100, 10

    #
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # 무작위로 가중치를 초기화합니다.
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(2000):
        # Forward pass : compute predicted y
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        #compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
        if t%100 == 99:
            print(t, loss)

        # backprop to compute gradients of a, b, ,c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_c = (grad_y_pred * x ** 3).sum()

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

def test_auto_grad():

    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # 무작위로 가중치를 초기화합니다.
    a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6

    for t in range(2000):
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())

        loss.backward()

        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')


