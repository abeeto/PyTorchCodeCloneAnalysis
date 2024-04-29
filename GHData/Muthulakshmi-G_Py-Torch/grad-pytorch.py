w=Variable(torch.Tensor([1.0]),requires_grad=True)

def forward(x):
    return x*w

def loss(x,y):
    y_pred=forward(x)
    return(y_pred-y)*(y_pred-y)


for epoch in range(10):

    for x_val,y_val in zip(x_data,y_data):
        l=loss(x_val,y_val)
        l.backward()
        print("\tgrad:",x_val,y_val,w.grad.data[0])

        w.data=w.data-0.01*w.grad.data

        w.grad.data.zero_()

    print("prograss:",epoch,l.data[0])
