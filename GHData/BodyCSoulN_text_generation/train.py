import math
import torch
from torch import nn
from data_preprocess import trans_dim
from d2l import torch as d2l

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # 梯度裁剪，比较模型的所有层的所有参数的梯度的l2长度和theta的大小，如果大，那就拉到theta，否则不动
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for p in params:
            p.grad[:] = theta / norm * p.grad

def train_epoch(model, loss, updater, train_iter, device):
    state = None
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state = model.module.begin_state(X.shape[0], device)
        else:
            state = model.begin_state(X.shape[0], device)
        # 变成一个1维的向量
        y = Y.T.reshape(-1)
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_hat, state = model(X, state)
        l = loss(y_hat, y.long()).mean()
        updater.zero_grad()
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        updater.step()
        metric.add(l * y.numel(), y.numel(), l)
    return math.exp(metric[0] / metric[1]), l

def train_epoch_data_parallel_1(model, loss, updater, train_iter, device):
    state = None
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if isinstance(model, torch.nn.DataParallel):
            state = model.module.begin_state(X.shape[0], device)
        else:
            state = model.begin_state(X.shape[0], device)
        y = Y.T.reshape(-1)
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        X = X.T
        y_hat, state = model(X, state)
        y_hat = y_hat.T
        l = loss(y_hat, y.long()).mean()
        updater.zero_grad()
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        updater.step()
        metric.add(l * y.numel(), y.numel(), l)
    return math.exp(metric[0] / metric[1]), l
    
def train_epoch_data_parallel_0(model, loss, updater, train_iter, device):
    state = None
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if isinstance(model, torch.nn.DataParallel):
            state = model.module.begin_state(X.shape[0], device)
        else:
            state = model.begin_state(X.shape[0], device)
        y = Y.T.reshape(-1)
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        state = trans_dim(state)
        y_hat, state = model(X, state)
        state = trans_dim(state)
        l = loss(y_hat, y.long()).mean()
        updater.zero_grad()
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        updater.step()
        metric.add(l * y.numel(), y.numel(), l)
    return math.exp(metric[0] / metric[1]), l     
    
def train_novel(model, local_rank, train_iter, lr, num_epochs, device):
    model.train()
    updater = torch.optim.Adam(model.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        ppl, l = train_epoch(model, loss, updater, train_iter, device)
        timer.stop()
        if epoch % 10 == 0:
            print(f'loss: {l:.2f}, training time per epoch: {timer.avg():.2f}',
                f'in device {str(local_rank)}')
    print(f'困惑度 {ppl:.1f}')
    # print(predict('叶凡'))
    
def predict_novel(prefix, num_preds, model, vocab, device):
    model.eval()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state = model.module.begin_state(1, device)
    else:
        state = model.begin_state(1, device)
    outputs = [vocab[prefix[0]]]
    get_inputs = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    
    for y in prefix[1:]:
        # state = trans_dim(state)
        _, state = model(get_inputs(), state)
        # state = trans_dim(state)
        outputs.append(vocab[y])
        
    for i in range(num_preds):
        # state = trans_dim(state)
        y, state = model(get_inputs(), state)
        # state = trans_dim(state)
        outputs.append(y.argmax(dim=1))
    return ''.join([vocab.to_token(index) for index in outputs])