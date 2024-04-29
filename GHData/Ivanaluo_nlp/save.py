torch.save(net1, 'net.pkl')  # 保存整个网络
torch.save(net1.state_dict(), 'net_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)


def restore_net():   # 这种方式将会提取整个神经网络, 网络大的时候可能会比较慢.
    # restore entire net1 to net2
    net2 = torch.load('net.pkl')
    prediction = net2(x)

def restore_params():   # 这种方式将会提取所有的参数, 然后再放到你的新建网络中.
    # 新建 net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # 将保存的参数复制到 net3
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)