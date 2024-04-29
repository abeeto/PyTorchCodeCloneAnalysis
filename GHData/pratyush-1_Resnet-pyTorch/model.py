def resnet18():
    layers=[2, 2,2, 2]
    net = ResNet(BasicBlock, layers)
    return net
    
net=resnet18()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)  